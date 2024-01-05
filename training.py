import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def logit_prediction_loss(params, apply_fn, x_batched, y_batched):
    def loss_fn(x, y):
        logits = apply_fn(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y)

    return jax.vmap(loss_fn)(x_batched, y_batched).mean()


def signal_mse_loss(params, apply_fn, x_batched, y_batched):
    def loss_fn(x, y):
        out = apply_fn(params, x)
        return optax.l2_loss(out, y)

    return jax.vmap(loss_fn)(x_batched, y_batched).mean()


def optimization_step(state: TrainState, x_batched: jax.Array, y_batched: jax.Array, loss_fn):
    loss_value, loss_gradients = jax.value_and_grad(loss_fn)(
        state.params, state.apply_fn, x_batched, y_batched
    )
    state = state.apply_gradients(grads=loss_gradients)
    return state, loss_value
