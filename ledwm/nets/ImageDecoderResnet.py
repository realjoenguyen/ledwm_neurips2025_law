from termcolor import cprint
from ledwm import jaxutils, ninjax as nj
from ledwm.nets.Conv2D import Conv2D
from ledwm.nets.ImageEncoderResnet import NORMALIZE_THRESHOLD
from ledwm.nets.Linear import LinearAct

import flax.linen as nn

import jax
import jax.numpy as jnp
import numpy as np

from ledwm.nets.MLP import MLP


class ImageDecoderResnet(nj.Module):
    def __init__(
        self,
        shape,
        depth=96,
        blocks=0,
        resize="stride",
        minres=3,
        sigmoid=False,
        kernel=4,
        stride=2,
        kernels=(),
        strides=(),
        stages=0,
        input2cnn=False,
        depths=(),
        **kw,
    ):
        self._shape = shape
        self._stages = stages
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._sigmoid = sigmoid
        self._kernel = kernel
        self._stride = stride
        self.kernels = kernels
        assert not isinstance(self.kernels, int), f"{self.kernels=}"
        self.strides = strides
        assert not isinstance(self.strides, int), f"{self.strides=}"
        # assert len(self._kernels) == len(self._strides)
        self.input2cnn_mlp = input2cnn
        self._kw = kw
        self.depths = depths
        print(
            f"image_decoder.config | resize={self._resize} | kernel={self._kernel} | "
            f"stride={self._stride} | min_resolution={self._minres} | "
            f"kernels={self.kernels} | strides={self.strides}"
        )

    def __call__(self, x):  # (bs*bl, d_stoch**2 + d_deter)
        if self._stages == 0:
            stages = int(
                np.log2(self._shape[-2]) - np.log2(self._minres)
            )  # 2**4 = 16 - 2**2 = 4  = 2
        else:
            stages = self._stages
            assert len(self.kernels) == self._stages

        print(f"image_decoder.config | stages={stages}")
        if len(self.depths) == 0:
            depth = self._depth * 2 ** (stages - 1)  # 96 * 2**(2-1) = 96 * 2**1 = 192
        else:
            depth = self.depths[0]

        x = jaxutils.cast_to_compute(x)
        # x: dim -> minres * minres * depth -> reshape
        print(f"image_decoder.tensor | name=input | shape={x.shape}")
        # dropout

        if self.input2cnn_mlp:
            x = self.get(
                "in",
                MLP,
                shape=None,
                layers=1,
                units=self._minres * self._minres * depth,
                norm="layer",
                act="silu",
                dist="none",
            )(x)
            x = x.reshape(-1, self._minres, self._minres, depth)
            print(
                f"image_decoder.tensor | name=cnn_input | source=mlp | "
                f"shape={x.shape}"
            )
        else:
            x = self.get("in", LinearAct, (self._minres, self._minres, depth))(x)
            print(
                f"image_decoder.tensor | name=cnn_input | source=linear | "
                f"shape={x.shape}"
            )

        # convert from d_in -> depth: (bs*bl, depth)
        for i in range(stages):
            # for j in range(self._blocks):
            #     # skip = x
            #     kw = {**self._kw, "preact": True}
            #     x = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw)(x)
            #     x = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw)(x)
            # x += skip
            # print(x.shape)

            depth //= 2
            kw = {**self._kw, "preact": False}
            if i == stages - 1:
                # kw = {"bias_for_last": True}  # kw for the last layer
                kw = {}
                depth = self._shape[-1]

            if self._resize == "stride":
                # each time double the size of x + reduce the depth. if the last iteration, -> depth of the output: 192 -> 192 / 2 = 96 / 2 = 48
                if len(self.kernels) > 0:
                    kernel = self.kernels[i]
                else:
                    kernel = self._kernel
                if len(self.strides) > 0:
                    stride = self.strides[i]
                else:
                    stride = self._stride
                if len(self.depths) > 0:
                    depth = self.depths[i + 1]  # first is input

                x = self.get(
                    f"s{i}res",
                    Conv2D,
                    depth,
                    kernel,
                    stride,
                    transp=True,
                    **kw,
                )(x)
                print(
                    f"image_decoder.stage | index={i} | shape={x.shape} | "
                    f"kernel={kernel} | stride={stride} | options={kw}"
                )

            # elif self._resize == "stride3":
            # s = 3 if i == stages - 1 else 2
            # k = 5 if i == stages - 1 else 4
            # x = self.get(f"s{i}res", Conv2D, depth, k, s, transp=True, **kw)(x)

            # elif self._resize == "resize":
            #     x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
            #     x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
            else:
                raise NotImplementedError(self._resize)

        if max(x.shape[1:-1]) > max(self._shape[:-1]):
            padh = (x.shape[1] - self._shape[0]) / 2
            padw = (x.shape[2] - self._shape[1]) / 2
            x = x[:, int(np.ceil(padh)) : -int(padh), :]
            x = x[:, :, int(np.ceil(padw)) : -int(padw)]

        # linear before output
        # collapse the last 3 dims
        x = x.reshape(-1, np.prod(x.shape[-3:]))
        x = self.get("out", LinearAct, self._shape, bias_last=True)(x)  # act='none'
        print(f"image_decoder.tensor | name=linear_output | shape={x.shape}")
        assert x.shape[-3:] == self._shape, (
            x.shape,
            self._shape,
        )  # self._shape is from obs_space
        print(
            f"image_decoder.tensor | name=output | shape={x.shape} | "
            f"expected_shape={self._shape}"
        )

        if self._sigmoid:
            cprint("image_decoder.config | output_activation=sigmoid", "yellow")
            x = jax.nn.sigmoid(x)
        # else:
        #     x += NORMALIZE_THRESHOLD  # because it use layernorm
        return x
