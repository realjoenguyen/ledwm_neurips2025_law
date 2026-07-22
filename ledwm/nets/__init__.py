import jax
import jax.numpy as jnp
import ledwm.tfp_compat  # noqa: F401
from tensorflow_probability.substrates.jax import distributions as tfd
from ledwm import jaxutils

f32 = jnp.float32
f16 = jnp.float16


def get_act(name):
    if callable(name):
        return name
    elif name == "none":
        return lambda x: x
    elif name == "mish":
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    elif name == "gelu2":
        return lambda x: jax.nn.sigmoid(1.702 * x) * x
    elif hasattr(jax.nn, name):
        return getattr(jax.nn, name)
    else:
        raise NotImplementedError(name)
