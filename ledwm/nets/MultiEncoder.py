#   encoder: {mlp_keys: '.*', cnn_keys: '.*', act: silu, norm: layer, mlp_layers: 5, mlp_units: 1024, cnn: resnet, cnn_depth: 96, cnn_blocks: 0, resize: stride, winit: normal, fan: avg}
from termcolor import cprint
from ledwm import jaxutils, ninjax as nj
from ledwm.nets.ImageEncoderResnet import ImageEncoderResnet
from ledwm.nets.MLP import MLP
from ledwm.nets import f32
import jax.numpy as jnp
import re


class MultiEncoder(nj.Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        depth=96,
        blocks=0,
        depths=[],
        # cnn_minres=4,
        stages=2,
        kernels=[],
        resize="stride",
        kernel=4,
        stride=2,
        symlog_inputs=False,
        minres=4,
        sent=False,
        image_shape=[16, 16],
        small_image=False,
        **kw,
    ):
        self.small_image = small_image
        excluded = ("is_first", "is_last")
        shapes = {
            k: v
            for k, v in shapes.items()
            if (k not in excluded and not k.startswith("log_"))
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if (len(v) == 3 and re.match(cnn_keys, k))
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if (len(v) in (1, 2) and re.match(mlp_keys, k))
        }
        assert not (
            "token" in self.mlp_shapes and "token_embed" in self.mlp_shapes
        ), "Probably shouldn't have both token and token_embed, use token$?"
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print(f"multi_encoder.inputs | type=cnn | shapes={self.cnn_shapes}")
        cprint(f"multi_encoder.image_reshape | shape={image_shape}", "red")
        print(f"multi_encoder.inputs | type=mlp | shapes={self.mlp_shapes}")
        cnn_kw = {
            **kw,
            "minres": minres,
            "name": "cnn",
            "kernel": kernel,
            "stride": stride,
            "kernels": kernels,
            "stages": stages,
            "resize": resize,
            "depths": depths,
            # "shortcut": shortcut,
        }
        mlp_kw = {**kw, "symlog_inputs": symlog_inputs, "name": "mlp"}
        self.image_shape = image_shape

        if cnn == "resnet":
            self._cnn = ImageEncoderResnet(blocks, **cnn_kw)
        else:
            raise NotImplementedError(cnn)
        if self.mlp_shapes:
            self._mlp = MLP(None, mlp_layers, mlp_units, dist="none", **mlp_kw)
        self.preprocessors = {}

    def __call__(self, data, zero_mlp=False, zero_cnn=False):
        # some_key = image, some_shape = (16, 16, 17)
        some_key, some_shape = list(self.shapes.items())[0]
        batch_dims = data[some_key].shape[: -len(some_shape)]  # (bs, bl)
        data = {
            k: v.reshape((-1,) + v.shape[len(batch_dims) :]) for k, v in data.items()
        }
        outputs = []

        if self.cnn_shapes:
            # (bs * bl, 16, 16, 17)
            inputs = jnp.concatenate(
                [
                    data[k][:, : self.image_shape[0], : self.image_shape[1], :]
                    for k in self.cnn_shapes
                ],
                -1,
            )
            output = self._cnn.__call__(inputs)  # (bs*bl, cnn_dim=4*4*96*2=3092)
            output = output.reshape((output.shape[0], -1))
            if zero_cnn:
                output = jnp.zeros_like(output)
            # (bs * bl, d_cnn)
            print(f"multi_encoder.tensor | name=cnn_output | shape={output.shape}")
            outputs.append(output)

        if self.mlp_shapes:
            # (bs*bl, token_dim=512) lt
            inputs = [
                data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
                for k in self.mlp_shapes
            ]

            inputs = jnp.concatenate([x.astype(f32) for x in inputs], -1)
            inputs = jaxutils.cast_to_compute(inputs)  # (token_dim, bs*bl)
            output = self._mlp(inputs)  # (bs*bl, mlp_dim=1024)
            if zero_mlp:
                # from jax.jax._src.basearray import ArrayLike
                # assert isinstance(output, ArrayLike)
                output = jnp.zeros_like(output)
            # bs * bl, mlp_dim
            print(f"multi_encoder.tensor | name=mlp_output | shape={output.shape}")
            outputs.append(output)

        outputs = jnp.concatenate(outputs, -1)
        # bs, bl, mlp_dim + cnn_dim
        outputs = outputs.reshape(batch_dims + outputs.shape[1:])
        print(f"multi_encoder.tensor | name=output | shape={outputs.shape}")

        return outputs
