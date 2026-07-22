from ledwm import ninjax as nj
from ledwm.nets.Linear import LinearAct
from ledwm.nets.Norm import Norm
from ledwm.jaxutils import cast_to_compute


import jax
import jax.numpy as jnp


class GRUCell(nj.Module):
    def __init__(self, units, norm, winit, layer_id):
        self.units = units
        self.norm = norm
        self.winit = winit
        self.layer_id = layer_id

    def __call__(self, deter, x):
        # norm input
        t = self.layer_id
        x_norm = self.get(f"norm_input_{t}", Norm, self.norm)(x)

        # update gate
        update_input = self.get(
            f"update_input_x_{t}",
            LinearAct,
            units=self.units,
            winit="normal",
        )(x_norm) + self.get(
            f"update_input_h_{t}",
            LinearAct,
            units=self.units,
            winit="ortho",
        )(deter)
        update_gate = jax.nn.sigmoid(update_input)

        # reset gate
        reset_input = self.get(
            f"reset_input_x_{t}",
            LinearAct,
            units=self.units,
            winit="normal",
        )(x_norm) + self.get(
            f"reset_input_h_{t}",
            LinearAct,
            units=self.units,
            winit="ortho",
        )(deter)
        reset_gate = jax.nn.sigmoid(reset_input)

        # new hidden state
        new_input = self.get(
            f"new_input_x_{t}",
            LinearAct,
            units=self.units,
            winit="normal",
        )(x_norm) + self.get(
            f"new_input_h_{t}",
            LinearAct,
            units=self.units,
            winit="ortho",
        )(reset_gate * deter)
        new_info = jnp.tanh(new_input)

        new_deter = (1 - update_gate) * deter + update_gate * new_info
        return {"deter": cast_to_compute(new_deter)}
