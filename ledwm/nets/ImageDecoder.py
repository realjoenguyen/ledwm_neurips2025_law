# from ledwm import jaxutils, ninjax as nj
# from ledwm.nets.Conv2D import Conv2D
# from ledwm.nets.ImageEncoderResnet import NORMALIZE_THRESHOLD
# from ledwm.nets.Linear import Linear


# import jax
# import jax.numpy as jnp
# import numpy as np


# class ImageDecoderResnet(nj.Module):
#     def __init__(
#         self,
#         shape,
#         depth,
#         blocks,
#         resize,
#         minres,
#         sigmoid,
#         kernel=4,
#         stride=2,
#         kernels=[],
#         strides=[],
#         stages=0,
#         **kw,
#     ):
#         self._shape = shape
#         self._stages = stages
#         self._depth = depth
#         self._blocks = blocks
#         self._resize = resize
#         self._minres = minres
#         self._sigmoid = sigmoid
#         self._kernel = kernel
#         self._stride = stride
#         self._kernels = kernels
#         self._strides = strides
#         # assert len(self._kernels) == len(self._strides)
#         self._kw = kw
#         print(
#             f"ImageDecoderResnet {self._resize=}, {self._kernel=}, {self._stride=}, {self._minres=} "
#         )

#     def __call__(self, x):  # (bs*bl, d_stoch**2 + d_deter)
#         if self._stages == 0:
#             stages = int(
#                 np.log2(self._shape[-2]) - np.log2(self._minres)
#             )  # 2**4 = 16 - 2**2 = 4  = 2
#         else:
#             stages = self._stages
#             assert len(self._kernels) == self._stages
#         print("Decoder Stages:", stages)

#         depth = self._depth * 2 ** (stages - 1)  # 96 * 2**(2-1) = 96 * 2**1 = 192
#         x = jaxutils.cast_to_compute(x)
#         # x: dim -> minres * minres * depth -> reshape
#         print("Decoder Input:", x.shape)
#         x = self.get("in", Linear, (self._minres, self._minres, depth))(x)
#         print(f"After Linear: {x.shape}")
#         # convert from d_in -> depth: (bs*bl, depth)

#         for i in range(stages):
#             depth //= 2
#             kw = {**self._kw, "preact": False}
#             if i == stages - 1:
#                 kw = {}
#                 depth = self._shape[-1]

#             # each time double the size of x + reduce the depth. if the last iteration, -> depth of the output: 192 -> 192 / 2 = 96 / 2 = 48
#             if len(self._kernels) > 0:
#                 x = self.get(
#                     f"s{i}res",
#                     Conv2D,
#                     depth,
#                     self._kernels[i],
#                     # self._strides[i],
#                     self._stride,
#                     transp=True,
#                     **kw,
#                 )(x)
#             else:
#                 x = self.get(
#                     f"s{i}res",
#                     Conv2D,
#                     depth,
#                     self._kernel,
#                     self._stride,
#                     transp=True,
#                     **kw,
#                 )(x)
#             print(f"Decoder Stage {i}: {x.shape}")
