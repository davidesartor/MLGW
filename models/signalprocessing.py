import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class SequenceEmbedding(nn.Module):
    vocab_size: int
    embedding_dim: int
    max_context_len: int

    @nn.compact
    def __call__(self, x: jax.Array):
        pos = jnp.arange(len(x))[::-1]
        token_embedding = nn.Embed(self.vocab_size, self.embedding_dim)(x)
        position_embedding = nn.Embed(self.max_context_len, self.embedding_dim)(pos)
        return token_embedding + position_embedding


class AttentionHead(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jax.Array):
        keys = nn.Dense(self.channels, use_bias=False)(x)
        values = nn.Dense(self.channels, use_bias=False)(x)
        queries = nn.Dense(self.channels, use_bias=False)(x)

        causal_mask = jnp.tril(jnp.ones((len(x), len(x))))
        weights: jax.Array = keys @ queries.T / jnp.sqrt(self.channels)
        weights = jax.nn.softmax(jnp.where(causal_mask == 1, weights, -jnp.inf), axis=-1)
        return weights @ values


class MultiAttentionHead(nn.Module):
    channels: int
    n_heads: int = 1

    @nn.compact
    def __call__(self, x: jax.Array):
        return jnp.concatenate(
            [AttentionHead(self.channels)(x) for _ in range(self.n_heads)], axis=-1
        )


class TransformerBlock(nn.Module):
    head_size: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jax.Array):
        time, channels = x.shape
        x = x + nn.Dense(channels)(MultiAttentionHead(self.head_size, self.n_heads)(x))
        x = x + MLP(4 * channels, channels)(x)
        return x


class SSM(nn.Module):
    state_dim: int
    dt: float | None = None

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape
        A = self.param("A", lambda k: -(1 + jnp.arange(self.state_dim, dtype=jnp.float32)))

        # A = -nn.softplus(nn.Dense(self.state_dim, use_bias=False)(x))
        B = nn.Dense(self.state_dim, use_bias=False)(x)
        C = nn.Dense(self.state_dim, use_bias=False)(x)

        # discretization
        dt = self.dt or nn.softplus(nn.Dense(1, use_bias=True)(x))
        A, B = jnp.exp(A * dt), (jnp.exp(A * dt) - 1) * B / A

        def scan_fn(el1, el2):
            (A1, B1), (A2, B2) = el1, el2
            return A1 * A2, A1 * B2

        _, Ak_B = jax.lax.associative_scan(scan_fn, (A, B))
        x = jnp.einsum("ls,ls,lc->lc", C, Ak_B, x)
        return x


class MambaBlock(nn.Module):
    state_dim: int

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape
        gate = nn.Dense(2 * channels, use_bias=False)(x)
        gate = nn.swish(gate)

        x = nn.Dense(2 * channels, use_bias=False)(x)
        x = SSM(self.state_dim)(x)
        x = nn.swish(x)
        x = SSM(self.state_dim)(x)

        x = nn.Dense(channels, use_bias=False)(x * gate)
        return x


class LanguageModelMixin:
    vocab_size: int
    max_context_len: int

    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError

    def logits(self, context: jax.Array) -> jax.Array:
        """Compute logits for next token.

        Args:
            context (context_len,): Sequence of token idxs.

        Returns:
            logits (context_len, vocab_size): Logits for next token.
        """
        return self(context)

    def generate_token(self, context: jax.Array, rng_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Generate next token.

        Args:
            context (context_len,): Sequence of token idxs.
            rng_key (PRNGKey): Random key.

        Returns:
            next_token (): Next token.
            updated_context (updated_context_len,): Updated context given the generated token.
        """
        logits = self.logits(context)[-1]
        next_token = jax.random.categorical(rng_key, logits, axis=-1)
        context = self.update_context(context, next_token)
        return next_token, context

    def update_context(self, context: jax.Array, next_token: jax.Array) -> jax.Array:
        """Update context given the generated token, truncated to max_context_len.

        Args:
            context (context_len,): Sequence of token idxs.
            next_token (): Next token.

        Returns:
            updated_context (updated_context_len,): Updated context given the generated token.
        """
        if len(context) < self.max_context_len:
            context = jnp.concatenate([context, next_token[None]])
        else:
            context = context.at[:-1].set(context[1:])
            context = context.at[-1].set(next_token)
        if len(context) > self.max_context_len:
            context = context[-self.max_context_len :]
        return context


class BigramLM(nn.Module, LanguageModelMixin):
    vocab_size: int
    max_context_len: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tokens_one_hot = jax.nn.one_hot(x, self.vocab_size)
        logits = nn.Dense(self.vocab_size)(tokens_one_hot)
        return logits


class TransormerLM(nn.Module, LanguageModelMixin):
    vocab_size: int
    max_context_len: int
    embedding_dim: int
    head_size: int
    n_heads: int = 1
    n_layers: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = SequenceEmbedding(self.vocab_size, self.embedding_dim, self.max_context_len)(x)
        for _ in range(self.n_layers):
            x = TransformerBlock(self.head_size, self.n_heads)(x)
        x = nn.Dense(self.vocab_size)(x)
        return x


class MambaLM(nn.Module, LanguageModelMixin):
    vocab_size: int
    max_context_len: int
    embedding_dim: int
    state_dim: int
    n_layers: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = SequenceEmbedding(self.vocab_size, self.embedding_dim, self.max_context_len)(x)
        for _ in range(self.n_layers):
            x = nn.RMSNorm()(x + MambaBlock(self.state_dim)(x))
        x = nn.Dense(self.vocab_size)(x)
        return x
