from ledwm import ninjax as nj
from ledwm.nets import Dist
from ledwm.nets.ImageDecoderResnet import ImageDecoderResnet
from ledwm.nets.Input import Input
from ledwm.nets.MLP import MLP
from ledwm.nets import f32, tfd


import jax.numpy as jnp
import numpy as np


import re


class MultiDecoder(nj.Module):
    def __init__(
        self,
        shapes,  # shape from obs_space of Messenger
        inputs=["tensor"],  # [deter, stoch]
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        depth=48,  # 96
        blocks=2,
        depths=[],
        image_dist="mse",
        vector_dist="mse",
        resize="stride",
        bins=255,
        outscale=1.0,
        minres=4,
        cnn_sigmoid=False,
        kernel=4,
        stride=2,
        # shortcut=False,
        kernels=[],
        strides=[],
        stages=0,
        **kw,
    ):
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_sigmoid = cnn_sigmoid
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        }  # 'image': 16, 16, 17
        if mlp_keys != "":
            self.mlp_shapes = {
                k: v for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
            }  # 'token_embed': (d_token,)
        else:
            self.mlp_shapes = {}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}

        print(f"multi_decoder.outputs | type=cnn | shapes={self.cnn_shapes}")
        print(f"multi_decoder.outputs | type=mlp | shapes={self.mlp_shapes}")

        cnn_kw = {
            **kw,
            "minres": minres,
            "sigmoid": cnn_sigmoid,
            "kernel": kernel,
            "stride": stride,
            "kernels": kernels,
            "stages": stages,
            "strides": strides,
            "depths": depths,
            "resize": resize,
        }
        print(f"multi_decoder.config | component=cnn | options={cnn_kw}")
        mlp_kw = {**kw, "dist": vector_dist, "outscale": outscale, "bins": bins}

        if self.cnn_shapes:
            shapes = list(self.cnn_shapes.values())
            assert all(x[:-1] == shapes[0][:-1] for x in shapes)
            shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
            if cnn == "resnet":
                self._cnn = ImageDecoderResnet(shape, **cnn_kw, name="cnn")
            elif cnn == "style":
                # self._cnn = ImageDecoderStyle(
                # shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name="cnn"
                # )
                raise NotImplementedError

            else:
                raise NotImplementedError(cnn)

        if self.mlp_shapes:
            self._mlp = MLP(
                self.mlp_shapes, mlp_layers, mlp_units, **mlp_kw, name="mlp"
            )  # d_token,
        self._inputs = Input(inputs, dims="deter")
        self._image_dist = image_dist

    def __call__(self, inputs, drop_loss_indices=None):
        """
        inputs:  'deter', 'logit', 'stoch', 'embed'
        """
        # concat stoch + deter: bs, bl, d_stoch + d_deter
        features = self._inputs.__call__(inputs)
        dists = {}

        if self.cnn_shapes:
            feat = features
            if drop_loss_indices is not None:
                feat = feat[:, drop_loss_indices]
            flat = feat.reshape([-1, feat.shape[-1]])  # (bs * bl, d_cnn )
            output = self._cnn(flat)  # bs * bl, 16, 16, 17
            output = output.reshape(
                feat.shape[:-1] + output.shape[1:]
            )  # bs, bl, 16, 16, 17
            split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
            means = jnp.split(output, split_indices, -1)  # bs, bl, 16, 16, 17
            dists.update(
                {
                    key: Dist.make_image_dist(self._image_dist, mean)
                    for (key, shape), mean in zip(self.cnn_shapes.items(), means)
                }
            )  # {'image': tfd.Independent(tfd.Normal(mean, 1), 3)}

        if self.mlp_shapes:
            dists.update(self._mlp.__call__(features))
        return dists
