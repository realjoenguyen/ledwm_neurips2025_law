import re

from termcolor import cprint
import ledwm.nets.Dist
from ledwm.nets.ImageDecoderResnet import ImageDecoderResnet
from ledwm.nets.Input import Input
from ledwm.nets.MLP import MLP
from ledwm import ninjax as nj
from ledwm import jaxutils
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from ledwm import constants
f32 = jnp.float32

#   decoder:
#     {
#       mlp_keys: ".*",
#       cnn_keys: ".*",
#       act: silu,
#       norm: layer,
#       mlp_layers: 5,
#       mlp_units: 1024,
#       cnn: resnet,
#       cnn_depth: 96,
#       cnn_blocks: 0,
#       image_dist: mse,
#       vector_dist: symlog_mse,
#       inputs: [deter, stoch],
#       resize: stride,
#       winit: normal,
#       fan: avg,
#       outscale: 1.0,
#       minres: 4,
#       cnn_sigmoid: False,
#     }


class DecoderSent(nj.Module):
    def __init__(
        self,
        shapes,  # shape from obs_space of Messenger
        inputs=["tensor"],  # [deter, stoch]
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=5,
        mlp_units=1024,
        cnn="resnet",
        depth=96,  # 96
        blocks=0,
        image_dist="mse",
        vector_dist="symlog_mse",
        resize="stride",
        bins=255,
        outscale=1.0,
        minres=4,
        cnn_sigmoid=False,
        kernels=[],
        strides=[],
        kernel=4,
        stride=2,
        sent_dim=constants.SENT_DIM,
        stages=0,
        input2cnn=False,
        depths=[],
        enforce_num_pixels=False,
        task="s1",
        weighted_loss=False,
        focal=False,
        dropout=0,
        **kw,
    ):
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.task = task
        self.weighted_loss = weighted_loss
        self.focal = focal
        self.enforce_num_pixels = enforce_num_pixels
        if cnn_keys != "":
            self.cnn_shapes = {
                k: v for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
            }  # 16, 16, 17
        else:
            self.cnn_shapes = {}

        if mlp_keys != "":
            self.mlp_shapes = {mlp_keys: (sent_dim,)}
        else:
            self.mlp_shapes = {}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}

        print(f"sentence_decoder.outputs | type=cnn | shapes={self.cnn_shapes}")
        print(f"sentence_decoder.outputs | type=mlp | shapes={self.mlp_shapes}")
        self.cnn_sigmoid = cnn_sigmoid

        cnn_kw = {
            **kw,
            "minres": minres,
            "sigmoid": cnn_sigmoid,
            "kernels": kernels,
            "strides": strides,
            "kernel": kernel,
            "stride": stride,
            "stages": stages,
            "depths": depths,
            "input2cnn": input2cnn,
        }
        mlp_kw = {**kw, "dist": vector_dist, "outscale": outscale, "bins": bins}

        if self.cnn_shapes:
            shapes = list(self.cnn_shapes.values())
            assert all(x[:-1] == shapes[0][:-1] for x in shapes)
            shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)  # 12, 12, 17
            if cnn == "resnet":
                self._cnn = ImageDecoderResnet(
                    shape, depth, blocks, resize, **cnn_kw, name="cnn"
                )
            elif cnn == "style":
                # self._cnn = ImageDecoderStyle(
                # shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name="cnn"
                # )
                raise NotImplementedError

            else:
                raise NotImplementedError(cnn)

        if self.mlp_shapes:
            # (d_S)
            self._input2sent = MLP(
                self.mlp_shapes, mlp_layers, mlp_units, **mlp_kw, name="input2sent"
            )
        # take the len of deter: (bs, bl, d_deter) == 3 as the base
        self._inputs = Input(inputs, dims="deter")
        self._image_dist = image_dist
        self.preprocessors = {}
        self._drop = nj.FlaxModule(
            nn.Dropout,
            rate=dropout,
            name="drop",
        )

    def __call__(self, inputs, drop_loss_indices=None, step=None, training=True):
        """
        inputs:  {'deter', 'logit', 'stoch'}
        """
        # concat stoch + deter: (bs, bl, d_deter + d_stoch)
        features = self._inputs.__call__(inputs)
        features = jaxutils.apply_dropout_on(features, self._drop, training, step)
        dists = {}

        # image reconstruction
        if self.cnn_shapes:
            feat = features
            if drop_loss_indices is not None:
                feat = feat[:, drop_loss_indices]

            flat = feat.reshape([-1, feat.shape[-1]])  # (bs * bl, d_cnn)
            # if hasattr(self, "_input2cnn"):
            #     flat = self._input2cnn.__call__(flat)  # (bs * bl, d_cnn)
            output = self._cnn.__call__(flat)  # bs * bl, 16, 16, 17
            # bs, bl, 16, 16, 17
            output = output.reshape(feat.shape[:-1] + output.shape[1:])

            split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
            means = jnp.split(output, split_indices, -1)  # bs, bl, 16, 16, 17
            dists.update(
                {
                    key: ledwm.nets.Dist.make_image_dist(
                        self._image_dist,
                        mean,
                        self.task,
                        weighted=self.weighted_loss,
                        focal=self.focal,
                    )
                    for (key, shape), mean in zip(self.cnn_shapes.items(), means)
                }
            )

        # sent reconstruction
        if self.mlp_shapes:
            dists.update(self._input2sent.__call__(features))

        assert len(dists) > 0, "no dists in DecoderSent"
        cprint(f"sentence_decoder.distributions | count={len(dists)}", "green")
        for k, v in dists.items():
            print(f"sentence_decoder.distribution | name={k} | value={v}")
        return dists
