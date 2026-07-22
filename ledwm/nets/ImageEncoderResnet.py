from termcolor import cprint
from ledwm import jaxutils, ninjax as nj
from ledwm.nets.Conv2D import Conv2D
from ledwm.nets.Linear import LinearAct
from ledwm.nets import get_act
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

NORMALIZE_THRESHOLD = 0.5
MEAN = 0.002352941176470588
STD = 0.04845002419288002


# def normalize_image(image, dense=False, sent=False):
#     cprint(f"Normalize {image.shape=} with mean {MEAN} and std {STD}, {sent=}", "red")
#     if dense:
#         # res = (image - MEAN) / STD
#         raise ValueError("not implemented")
#     else:
#         res = (image - MEAN) / STD
#         # raise ValueError("not implemented")
#     # if sent:
#     #     res *= T5_STD
#     return res


class ImageEncoderResnet(nj.Module):
    def __init__(
        self,
        blocks,
        resize,
        minres,
        kernel=4,
        stride=2,
        depth=None,
        normalize=True,
        preconv=False,
        pre_depth=None,
        pre_kernel=None,
        pre_stride=None,
        preconv_on_origin=False,
        # shortcut=False,
        kernels=(),
        strides=(),
        stages=0,
        depths=(),
        # input2cnn=False,
        concat_image=True,
        dropout=0,
        **kw,
    ):
        self.preconv = preconv
        self.strides = strides
        assert not isinstance(self.strides, int), f"{self.strides=}"
        self.depths = depths
        self.stages = stages
        self.concat_image = concat_image
        self.preconv_on_origin = preconv_on_origin
        self._kernels = kernels
        assert not isinstance(self._kernels, int), f"{self._kernels=}"
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._kw = kw
        self._kernel = kernel
        self._stride = stride
        self._normalize = normalize
        self._pre_depth = pre_depth
        self._pre_kernel = pre_kernel
        self._pre_stride = pre_stride
        # self.input2cnn = input2cnn
        print(
            f"image_encoder.config | resize={self._resize} | kernel={self._kernel} | "
            f"stride={self._stride} | activation={self._kw['act']} | options={kw}"
        )
        cprint(
            f"image_encoder.preconv | depth={self._pre_depth} | "
            f"kernel={self._pre_kernel} | stride={self._pre_stride}"
        )
        self.dropout = dropout

    def __call__(self, x, raw_image_dim=0, training=True):
        if self.stages > 0:
            stages = self.stages
        else:
            stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))

        if self.depths is not None and len(self.depths) > 0:
            depth = self.depths[0]
        else:
            depth = self._depth

        x = jaxutils.cast_to_compute(x)
        print(f"image_encoder.tensor | name=input | shape={x.shape}")

        if self.preconv:
            assert self._pre_depth is not None
            kw = {**self._kw, "preact": False}
            print(f"image_encoder.preconv | options={kw}")
            if self.concat_image and not self.preconv_on_origin:
                assert raw_image_dim > 0
                raw_image = x[..., -raw_image_dim:]
                x = x[..., :-raw_image_dim]
                print(
                    f"image_encoder.preconv | preserve_raw_image=true | "
                    f"input_shape={x.shape}"
                )
                x = self.get(
                    f"pres",
                    Conv2D,
                    self._pre_depth,
                    self._pre_kernel,
                    self._pre_stride,
                    **kw,
                )(x)
                x = jnp.concatenate([x, raw_image], axis=-1)
            else:
                x = self.get(
                    f"pres",
                    Conv2D,
                    self._pre_depth,
                    self._pre_kernel,
                    self._pre_stride,
                    **kw,
                )(x)
            print(f"image_encoder.tensor | name=preconv_output | shape={x.shape}")
        # else:
        # if self.input2cnn:
        #     x = self.get(
        #         "input2cnn", LinearAct, x.shape[-1], act="silu", norm="layer"
        #     )(x)
        #     print("Encoder: input2cnn", x.shape)

        for i in range(stages):
            kw = {**self._kw, "preact": False}
            if self._resize == "stride":
                if len(self._kernels) > 0:
                    kernel = self._kernels[i]
                else:
                    kernel = self._kernel

                if len(self.strides) > 0:
                    stride = self.strides[i]
                else:
                    stride = self._stride

                if len(self.depths) > 0:
                    depth = self.depths[i + 1]  # first is input

                kw["norm"] = "none"
                x = self.get(f"s{i}res", Conv2D, depth, kernel, stride, **kw)(x)
                # print("Dropout2D: ", x.shape, self.dropout, training)
                # x = jaxutils.dropout2d(x, self.dropout, nj.rng(), training=training)
                print(
                    f"image_encoder.stage | index={i} | shape={x.shape} | "
                    f"kernel={kernel} | stride={stride} | options={kw}"
                )

            # elif self._resize == "stride3":
            #     s = 2 if i else 3
            #     k = 5 if i else 4
            #     x = self.get(f"s{i}res", Conv2D, depth, k, s, **kw)(x)

            # elif self._resize == "mean":
            #     N, H, W, D = x.shape
            #     x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
            #     x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)

            # elif self._resize == "max":
            #     x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
            #     x = jax.lax.reduce_window(
            #         x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "same"
            #     )
            else:
                raise NotImplementedError(self._resize)

        #     for j in range(self._blocks):
        #         skip = x
        #         kw = {**self._kw, "preact": True}
        #         x = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw)(x)
        #         x = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw)(x)
        #         x += skip

        #     depth *= 2
        # if self._blocks:
        #     x = get_act(self._kw["act"])(x)

        # x:(bs, h, w, channels)
        x = x.reshape((x.shape[0], -1))
        print(f"image_encoder.tensor | name=output | shape={x.shape}")
        return x
