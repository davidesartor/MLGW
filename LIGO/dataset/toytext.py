from functools import partial
import jax
import jax.numpy as jnp


class TextTokenizer:
    def __init__(self, fulltext: str):
        self.vocab = sorted(list(set(fulltext)))
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}

    def encode(self, text: str) -> jax.Array:
        return jnp.array([self.char2idx[c] for c in text])

    def decode(self, idxs: jax.Array) -> str:
        return "".join([self.vocab[i] for i in idxs])


class TextDataset:
    def __init__(self, data_path: str):
        with open(data_path, "r") as f:
            self.fulltext = f.read()
        self.tokenizer = TextTokenizer(self.fulltext)
        self.encoded_fulltext = self.tokenizer.encode(self.fulltext)

    def sample(self, length: int, rng_key: jax.Array):
        max_idx = len(self.fulltext) - length
        idx = jax.random.randint(rng_key, shape=(), minval=0, maxval=max_idx)
        data = self.encoded_fulltext[idx + jnp.arange(length)]
        return data

    def get_batch(self, batch_size: int, context_len: int, rng_key: jax.Array):
        rng_keys = jax.random.split(rng_key, batch_size)
        data = jax.vmap(partial(self.sample, context_len + 1))(rng_keys)
        return data[:, :-1], data[:, 1:]
