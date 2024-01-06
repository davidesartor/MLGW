import jax
import jax.numpy as jnp
import flax.linen as nn
from .transformer import TransformerBlock
from .mamba import MambaBlock


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
            x = MambaBlock(self.state_dim, sample_rate=1)(x)
        x = nn.Dense(self.vocab_size)(x)
        return x
