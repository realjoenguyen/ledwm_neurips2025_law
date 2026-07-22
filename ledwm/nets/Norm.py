from ledwm import ninjax as nj
from ledwm.nets import f32


import jax
import jax.numpy as jnp


class Norm(nj.Module):
    def __init__(
        self,
        impl,
        batch_dims=None,  # only for batch norm
    ):
        self._impl = impl
        self._batch_dims = batch_dims  # number of batch dimensions

    def __call__(self, x, style=None):
        dtype = x.dtype
        if self._impl == "none":
            return x

        elif self._impl == "rsm":
            x = x.astype(f32)
            scale = self.get("scale", jnp.ones, x.shape[-1], f32)
            var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
            x = x * jax.lax.rsqrt(var + 1e-5) * scale
            return x.astype(dtype)

        elif self._impl == "layer":
            x = x.astype(f32)
            x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
            if style is None:
                x *= self.get("scale", jnp.ones, x.shape[-1], f32)
                x += self.get("bias", jnp.zeros, x.shape[-1], f32)
            else:
                x *= style[0]
                x += style[1]
            return x.astype(dtype)

        elif self._impl == "batch":
            # Batch normalization implementation with flexible batch_dims
            x = x.astype(f32)
            assert self._batch_dims is not None
            norm_axes = tuple(range(self._batch_dims))
            # Dynamically determine batch dimensions
            mean = jnp.mean(x, axis=norm_axes, keepdims=True)
            var = jnp.var(x, axis=norm_axes, keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)  # Normalize

            if style is None:
                scale = self.get("scale", jnp.ones, x.shape[-1], f32)
                bias = self.get("bias", jnp.zeros, x.shape[-1], f32)
            else:
                scale, bias = style

            x = x * scale + bias
            return x.astype(dtype)

        else:
            raise NotImplementedError(self._impl)
