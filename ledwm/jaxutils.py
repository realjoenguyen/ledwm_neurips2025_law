# %%
from calendar import c
from jax import random
import jax.numpy as jnp
import jax
from matplotlib import pyplot as plt
import numpy as np
import optax
import optree
from sklearn.metrics import f1_score, precision_recall_curve
from termcolor import cprint
from jax import lax
from einops import rearrange, repeat
from ledwm import ninjax as nj

tree_map = jax.tree_util.tree_map
tree_leaves = jax.tree_util.tree_leaves


def sg(x):
    return tree_map(jax.lax.stop_gradient, x)


COMPUTE_DTYPE = jnp.float16


def apply_dropout_on(val, fn_drop, training, step=None):
    """
    if training = True -> apply based on step
    if training = False -> not apply dropout
    """
    if training:
        val_drop = fn_drop(val, deterministic=False, rng=nj.rng())

        if step is not None:
            if step.shape[0] != val.shape[0]:
                step = step[0, None].repeat(val.shape[0], axis=0).reshape(-1)

            step_broadcast = step.reshape(step.shape[0], *([1] * (len(val.shape) - 1)))
            return jnp.where(step_broadcast == -1, val, val_drop)
        else:
            return val_drop
    else:
        return val


def apply_zero_deter(val, training, step=None):
    if training:
        if step is not None:
            if step.shape[0] != val.shape[0]:
                step = step[0, None].repeat(val.shape[0], axis=0).reshape(-1)
            step_broadcast = step.reshape(step.shape[0], *([1] * (len(val.shape) - 1)))
            return jnp.where(step_broadcast == -1, val, jnp.zeros_like(val))
        else:
            return jnp.zeros_like(val)
    else:
        return val


def flip_horizontally(entity_pos, avatar_pos):
    MAX = 10
    entity_pos = entity_pos.at[:, :, 0].set(MAX - entity_pos[:, :, 0])
    avatar_pos = avatar_pos.at[:, :, 0].set(MAX - avatar_pos[:, :, 0])
    return entity_pos, avatar_pos


def flip_vertically(entity_pos, avatar_pos):
    MAX = 10
    entity_pos = entity_pos.at[:, :, 1].set(MAX - entity_pos[:, :, 1])
    avatar_pos = avatar_pos.at[:, :, 1].set(MAX - avatar_pos[:, :, 1])
    return entity_pos, avatar_pos


def combined_flip(entity_pos, avatar_pos):
    epos, apos = flip_horizontally(entity_pos, avatar_pos)
    return flip_vertically(epos, apos)


def no_flip(entity_pos, avatar_pos):
    return entity_pos, avatar_pos


def random_flip(entity_pos, avatar_pos, rng):
    # Produce a random number and categorize into four ranges
    rand_num = jax.random.uniform(rng)
    index = jnp.digitize(rand_num, bins=jnp.array([0.25, 0.5, 0.75]))

    # Define the branches as a list of functions
    branches = [
        no_flip,  # 0: No flip
        flip_horizontally,  # 1: Horizontal flip
        flip_vertically,  # 2: Vertical flip
        combined_flip,  # 3: Combined flip (horizontal then vertical)
    ]

    # Use jax.lax.switch to select the appropriate action
    entity_pos, avatar_pos = jax.lax.switch(index, branches, *(entity_pos, avatar_pos))

    return entity_pos, avatar_pos


def load_partial_checkpoint_shape(varibs, state):
    ckpt_paths, ckpt_params, _ = optree.tree_flatten_with_path(state)
    ckpt_dict = {k: v for k, v in zip(ckpt_paths, ckpt_params)}
    not_load_vars = []
    load_vars = []
    loaded = optree.tree_map_with_path(
        lambda p, x: (
            ckpt_dict[p]
            if p in ckpt_dict and ckpt_dict[p].shape == x.shape
            # and "sent_embed" not in p
            else x
        ),
        varibs,
    )

    def add_not_load(p, x):
        if p not in ckpt_dict:
            not_load_vars.append(p)
        elif ckpt_dict[p].shape != x.shape:
            not_load_vars.append((p, x.shape, ckpt_dict[p].shape))
        else:
            load_vars.append(p)
        # if p not in loaded:
        #     not_load_vars.append((p, x.shape))

    optree.tree_map_with_path(
        lambda p, x: add_not_load(p, x),
        varibs,
    )

    cprint("Does not exist in ckpt: . Format(p, x.shape, ckpt_dict[p].shape)", "red")
    cprint("\n".join([str(x) for x in not_load_vars]), "red")
    # cprint("Keep:", "green")
    # cprint("\n".join([str(x) for x in load_vars]), "green")

    return loaded


def load_partial_checkpoint(orig, ck_state, load_key="", exclude_key=""):
    ckpt_paths, ckpt_params, _ = optree.tree_flatten_with_path(ck_state)
    ckpt_dict = {k: v for k, v in zip(ckpt_paths, ckpt_params)}
    ckpt_vars = []
    orig_vars = []

    def has_exclude_key(x):
        return exclude_key in x[0] if exclude_key != "" else False

    loaded = optree.tree_map_with_path(
        lambda p, x: (
            ckpt_dict[p]
            if p in ckpt_dict and load_key in p[0] and not has_exclude_key(p)
            else x
        ),
        orig,
    )
    optree.tree_map_with_path(
        lambda p, x: (
            ckpt_vars.append(p)
            if p in ckpt_dict and load_key in p[0] and not has_exclude_key(p)
            else orig_vars.append(p)
        ),
        orig,
    )  # just for logging which paths are loaded
    # print("Loaded from ckpt: ")
    # print("\n".join([str(x) for x in ckpt_vars]))
    return loaded


def cast_to_compute(values):
    return tree_map(lambda x: x.astype(COMPUTE_DTYPE), values)


def compact_rssm_recurrent_state(state, gru_multi_layers=False):
    """Keep only values consumed by the next RSSM posterior step."""
    deter_key = "deter_layers" if gru_multi_layers else "deter"
    return {key: state[key] for key in ("stoch", deter_key)}


def parallel():
    try:
        jax.lax.axis_index("i")
        return True
    except NameError:
        return False


# def tensorstats(tensor, prefix=None):
#     metrics = {
#         "mean": tensor.mean(),
#         "std": tensor.std(),
#         "mag": jnp.abs(tensor).max(),
#         "min": tensor.min(),
#         "max": tensor.max(),
#         "dist": subsample(tensor),
#     }
#     if prefix:
#         metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
#     return metrics


# tensorstats with mask
def tensorstats(tensor, prefix=None, mask=None):
    # only calculate stats for the masked values
    if mask is not None:
        metrics = {
            "mean": (tensor * mask).sum() / mask.sum(),
            "std": (tensor * mask).std(),
            "mag": jnp.abs(tensor * mask).max(),
            "min": (tensor * mask).min(),
            "max": (tensor * mask).max(),
            "dist": subsample(tensor * mask),
        }
    else:
        metrics = {
            "mean": tensor.mean(),
            "std": tensor.std(),
            "mag": jnp.abs(tensor).max(),
            "min": tensor.min(),
            "max": tensor.max(),
            "dist": subsample(tensor),
        }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def subsample(values, amount=1024):
    values = values.flatten()
    if len(values) > amount:
        values = jax.random.permutation(nj.rng(), values)[:amount]
    return values


def scan(fn, inputs, start, unroll=True, modify=False):
    # default config unroll=False
    """
    Applies a function `fn` to a sequence of inputs `inputs` in a cumulative manner,
    starting from an initial value `start`.

    Args:
        fn: The function to apply to each input. It should take two arguments: the current
            cumulative value and the current input value, and return a tuple containing
            the updated cumulative value and the output value.
        inputs: The sequence of inputs to apply the function to.
        start: The initial value to start the cumulative operation.
        unroll: Whether to manually unroll the loop or not. If set to False, the function
            will accumulate the outputs and return the final accumulated value. If set to
            True, the function will manually unroll the loop and return a list of outputs
            at each step.
        modify: Whether to modify the inputs in-place or not. If set to False, the function
            will create a copy of the inputs and modify the copy. If set to True, the
            function will modify the inputs in-place.

    Returns:
        The final cumulative value if `unroll` is False, or a list of outputs at each step
        if `unroll` is True.
    """

    def fn2(carry, inp):
        return (fn(carry, inp),) * 2  # return new_carry, out

    # if unroll == False, return out : accumulated out
    # if unroll == True, manual unroll for loop
    if not unroll:
        return nj.scan(fn2, start, inputs, modify=modify)[1]

    length = len(jax.tree_util.tree_leaves(inputs)[0])
    carrydef = jax.tree_util.tree_structure(start)
    carry = start
    outs = []

    for index in range(length):
        carry, out = fn2(carry, tree_map(lambda x: x[index], inputs))
        flat, treedef = jax.tree_util.tree_flatten(out)
        assert treedef == carrydef, (treedef, carrydef)
        outs.append(flat)

    outs = [jnp.stack([carry[i] for carry in outs], 0) for i in range(len(outs[0]))]
    return carrydef.unflatten(outs)


def scan_with_output(fn, inputs, start, unroll=True, modify=False):
    """Scan where the recurrent carry is smaller than the recorded output.

    Unlike :func:`scan`, ``fn`` returns ``(new_carry, output)``. This avoids
    feeding output-only tensors back through every recurrent step.
    """
    if not unroll:
        return nj.scan(fn, start, inputs, modify=modify)[1]

    length = len(jax.tree_util.tree_leaves(inputs)[0])
    carry = start
    outdef = None
    outs = []

    for index in range(length):
        carry, out = fn(carry, tree_map(lambda x: x[index], inputs))
        flat, treedef = jax.tree_util.tree_flatten(out)
        outdef = outdef or treedef
        assert treedef == outdef, (treedef, outdef)
        outs.append(flat)

    outs = [jnp.stack([out[i] for out in outs], 0) for i in range(len(outs[0]))]
    return outdef.unflatten(outs)


def symlog(x):
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def switch(is_first, init, prev):
    # pred: bs,
    # lhs = rhs: bs, d_deter
    assert init.shape == prev.shape, (is_first.shape, init.shape, prev.shape)
    while len(is_first.shape) < len(init.shape):
        is_first = is_first[..., None]
    return jnp.where(is_first, init, prev)


# def video_grid(video):
#     B, T, H, W, C = video.shape
#     return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


def video_from_image_model(value):
    # T, H, B * W, C - bl, h, bs*w, c
    from ledwm.embodied.envs.MessengerSent import MessengerSent

    bl, h, w_, c = value.shape
    bs = w_ // h
    w = h
    # T, H, B*W, C
    # value = value.reshape((-1, *value.shape[2:]))
    value = value.reshape((bl * bs, h, w, c))
    image_model = [MessengerSent.log_image(img) for img in value]
    image_model = np.stack(image_model).reshape(
        bl, 256, 256 * bs, 3
    )  # bs * bl, 256, 256, c
    # image_model = np.stack(image_model).reshape(
    #     (bs, bl, *value.shape[2:])
    # )
    return image_model


def balance_stats(
    dist,
    tgt,
    thres_low=0.0,
    thres_high=1.0,
    mask=None,
    # take_is_first_only=False,  # take the first bl only for finetune wm
    reward_values=None,
):
    # Values are NaN when there are no positives or negatives in the current
    # batch, which means they will be ignored when aggregating metrics via
    # np.nanmean() later, as they should.
    """
    dists: predicted
    target: ground truth
    thres = 0.1 for reward, 0.5 for continuity
    """
    assert thres_high >= thres_low, (thres_low, thres_high)
    assert thres_low >= 0, thres_low
    assert thres_high <= 1.5, thres_high

    if reward_values is not None:
        # tgt is onehot
        onehot_tgt = tgt
        tgt = onehot_to_float(tgt, reward_values).astype(jnp.float32)
        assert tgt.ndim == onehot_tgt.ndim - 1, (tgt.ndim, onehot_tgt.ndim)
    else:
        tgt = tgt.astype(jnp.float32)

    # threshold_low <= pos <= threshold_high
    tgt_pos_mask = ((tgt >= thres_low) & (tgt <= thres_high)).astype(jnp.float32)

    # -threshold_high <= neg <= -threshold_low
    tgt_neg_mask = ((tgt >= -thres_high) & (tgt <= -thres_low)).astype(jnp.float32)

    tgt_fine_pos = (tgt >= 0.01).astype(jnp.float32)
    tgt_zero = ((tgt > -0.01) & (tgt < 0.01)).astype(jnp.float32)
    tgt_fine_neg = (tgt <= -0.01).astype(jnp.float32)

    if mask is not None:
        # mask out zero with -1 -> doesn't count when find zeros in tgt_zero
        # tgt_zero = jnp.where(mask == 0, -1, tgt_zero)
        tgt_zero = tgt_zero * mask
        tgt_pos_mask = tgt_pos_mask * mask  # mask out zero
        tgt_neg_mask = tgt_neg_mask * mask  # mask out zero
        tgt_mask = mask
    else:
        tgt_mask = jnp.ones_like(tgt)

    if reward_values is not None:
        pred_mode = reward_values[dist.argmax()]
    else:
        pred_mode = dist.mode().astype(jnp.float32)

    # threshold_low <= pos <= threshold_high
    pred_pos = ((pred_mode >= thres_low) & (pred_mode <= thres_high)).astype(
        jnp.float32
    )
    # -threshold_high <= neg <= -threshold_low
    pred_neg = ((pred_mode >= -thres_high) & (pred_mode <= -thres_low)).astype(
        jnp.float32
    )
    # pred_pos_precision = (tgt_pos_mask * pred_pos).sum() / pred_pos.sum()
    pred_fine_pos = (pred_mode >= 0.01).astype(jnp.float32)
    # fine_pos_precision = (fine_pos_pred * fine_pos).sum() / fine_pos.sum()
    pred_zero = ((pred_mode > -0.01) & (pred_mode < 0.01)).astype(jnp.float32)
    # zero_pad_precision = (tgt_pos_mask * zero_pred).sum() / zero_pred.sum()
    pred_fine_neg = (pred_mode <= -0.01).astype(jnp.float32)

    if mask is not None:
        # mask out pred
        pred_zero = pred_zero * mask
        pred_pos = pred_pos * mask
        pred_neg = pred_neg * mask
        pred_fine_neg = pred_fine_neg * mask
        pred_fine_pos = pred_fine_pos * mask

    nonzero_mask = tgt_pos_mask + tgt_neg_mask
    if reward_values is not None:
        # onehot_tgt: bs, bl, n_class
        loss = -dist.log_prob(onehot_tgt)  # bs, bl
    else:
        loss = -dist.log_prob(tgt)
    res = {}

    if hasattr(dist, "kl"):
        kl = dist.kl(tgt)
        pos_kl = (kl * tgt_pos_mask).sum() / tgt_pos_mask.sum()
        neg_kl = (kl * tgt_neg_mask).sum() / tgt_neg_mask.sum()
        zero_kl = (kl * tgt_zero).sum() / tgt_zero.sum()
        # mse = (pred_mode - tgt) ** 2
        abs_diff = jnp.abs(pred_mode - tgt)
        pos_diff = (abs_diff * tgt_pos_mask).sum() / tgt_pos_mask.sum()
        # max
        pos_diff_max = (abs_diff * tgt_pos_mask).max()
        # min, fill pos_mask with inf to get min
        pos_diff_min = (abs_diff * tgt_pos_mask + (1 - tgt_pos_mask) * jnp.inf).min()
        neg_diff = (abs_diff * tgt_neg_mask).sum() / tgt_neg_mask.sum()
        neg_diff_max = (abs_diff * tgt_neg_mask).max()
        neg_diff_min = (abs_diff * tgt_neg_mask + (1 - tgt_neg_mask) * jnp.inf).min()
        zero_diff = (abs_diff * tgt_zero).sum() / tgt_zero.sum()
        res.update(
            dict(
                pos_kl=pos_kl,
                neg_kl=neg_kl,
                zero_kl=zero_kl,
                pos_diff=pos_diff,
                pos_diff_max=pos_diff_max,
                pos_diff_min=pos_diff_min,
                neg_diff=neg_diff,
                neg_diff_max=neg_diff_max,
                neg_diff_min=neg_diff_min,
                zero_diff=zero_diff,
                pos_kl_rate=(kl * tgt_pos_mask).sum() / kl.sum(),
                neg_kl_rate=(kl * tgt_neg_mask).sum() / kl.sum(),
            )
        )

    from ledwm.WM import MASK_VALUE_DIST

    res.update(
        dict(
            # NEG
            neg_loss=(loss * tgt_neg_mask).sum() / tgt_neg_mask.sum(),
            neg_mode=(pred_mode * tgt_neg_mask).sum() / tgt_neg_mask.sum(),
            pred_neg_dist=jnp.where(tgt_neg_mask, pred_mode, MASK_VALUE_DIST).reshape(
                -1
            ),
            data_dist=jnp.where(tgt_mask, tgt, MASK_VALUE_DIST).reshape(-1),
            # pred_dist=jnp.where(tgt_mask, pred_mode, MASK_VALUE_DIST).reshape(-1),
            pos_pred_dist=jnp.where(tgt_pos_mask, pred_mode, MASK_VALUE_DIST).reshape(
                -1
            ),
            neg_pred_dist=jnp.where(tgt_neg_mask, pred_mode, MASK_VALUE_DIST).reshape(
                -1
            ),
            data_neg_dist=jnp.where(tgt_neg_mask, tgt, MASK_VALUE_DIST).reshape(-1),
            # POS
            pos_loss=(loss * tgt_pos_mask).sum() / tgt_pos_mask.sum(),
            pos_mode=(pred_mode * tgt_pos_mask).sum() / tgt_pos_mask.sum(),
            pred_pos_dist=jnp.where(tgt_pos_mask, pred_mode, MASK_VALUE_DIST).reshape(
                -1
            ),
            data_pos_dist=jnp.where(tgt_pos_mask, tgt, MASK_VALUE_DIST).reshape(-1),
            pos_recall=(pred_pos * tgt_pos_mask).sum() / tgt_pos_mask.sum(),
            # TODO check alignment of tgt_pos_mask and pred_pos
            # TODO: check gt multi-step reward
            pos_count=tgt_pos_mask.sum(),
            neg_count=tgt_neg_mask.sum(),
            pos_precision=((tgt_pos_mask * pred_pos).sum() / pred_pos.sum()),
            neg_recall=(pred_neg * tgt_neg_mask).sum() / tgt_neg_mask.sum(),
            neg_precision=((tgt_neg_mask * pred_neg).sum() / pred_neg.sum()),
            fine_pos_recall=(pred_fine_pos * tgt_fine_pos).sum() / tgt_fine_pos.sum(),
            fine_pos_precision=(
                (tgt_fine_pos * pred_fine_pos).sum() / pred_fine_pos.sum()
            ),
            fine_neg_recall=(pred_fine_neg * tgt_fine_neg).sum() / tgt_fine_neg.sum(),
            fine_neg_precision=(
                (tgt_fine_neg * pred_fine_neg).sum() / pred_fine_neg.sum()
            ),
            zero_recall=(tgt_zero * pred_zero).sum() / tgt_zero.sum(),
            zero_precision=((tgt_zero * pred_zero).sum() / pred_zero.sum()),
            zero_loss=(loss * tgt_zero).sum() / tgt_zero.sum(),
            fine_pos_cnt=tgt_fine_pos.sum(),
            fine_neg_cnt=tgt_fine_neg.sum(),
            pos_rate=tgt_pos_mask.mean(),
            pos_rate_nonzero=tgt_pos_mask.sum() / nonzero_mask.sum(),
            neg_rate=tgt_neg_mask.mean(),
            neg_rate_nonzero=tgt_neg_mask.sum() / nonzero_mask.sum(),
            avg=tgt.astype(jnp.float32).mean(),
            mean=dist.mean().astype(jnp.float32).mean(),
        )
    )
    # if res has nan or inf due to / 0, set to 0
    # res = tree_map(lambda x: jnp.where(jnp.isnan(x) | jnp.isinf(x), 0, x), res)
    return res


class Moments(nj.Module):
    def __init__(
        self, impl="mean_std", decay=0.99, max=1e8, eps=0.0, perclo=5, perchi=95
    ):
        self.impl = impl
        self.decay = decay
        self.max = max
        self.eps = eps
        self.perclo = perclo
        self.perchi = perchi
        if self.impl == "off":
            pass
        elif self.impl == "mean_std":
            self.step = nj.Variable(jnp.zeros, (), jnp.int32, name="step")
            self.mean = nj.Variable(jnp.zeros, (), jnp.float32, name="mean")
            self.sqrs = nj.Variable(jnp.zeros, (), jnp.float32, name="sqrs")
        elif self.impl == "min_max":
            self.low = nj.Variable(jnp.zeros, (), jnp.float32, name="low")
            self.high = nj.Variable(jnp.zeros, (), jnp.float32, name="high")
        elif self.impl == "perc_ema":
            self.low = nj.Variable(jnp.zeros, (), jnp.float32, name="low")
            self.high = nj.Variable(jnp.zeros, (), jnp.float32, name="high")
        elif self.impl == "perc_ema_corr":
            self.step = nj.Variable(jnp.zeros, (), jnp.int32, name="step")
            self.low = nj.Variable(jnp.zeros, (), jnp.float32, name="low")
            self.high = nj.Variable(jnp.zeros, (), jnp.float32, name="high")
        elif self.impl == "mean_mag":
            self.mag = nj.Variable(jnp.zeros, (), jnp.float32, name="mag")
        elif self.impl == "max_mag":
            self.mag = nj.Variable(jnp.zeros, (), jnp.float32, name="mag")
        else:
            raise NotImplementedError(self.impl)

    def __call__(self, x):
        self.update(x)
        return self.stats()

    def update(self, x):
        if parallel():
            mean = lambda x: jax.lax.pmean(x.mean(), "i")
            min_ = lambda x: jax.lax.pmin(x.min(), "i")
            max_ = lambda x: jax.lax.pmax(x.max(), "i")
            per = lambda x, q: jnp.percentile(jax.lax.all_gather(x, "i"), q)
        else:
            mean = jnp.mean
            min_ = jnp.min
            max_ = jnp.max
            per = jnp.percentile

        x = sg(x.astype(jnp.float32))
        m = self.decay
        if self.impl == "off":
            pass

        elif self.impl == "mean_std":
            self.step.write(self.step.read() + 1)
            self.mean.write(m * self.mean.read() + (1 - m) * mean(x))
            self.sqrs.write(m * self.sqrs.read() + (1 - m) * mean(x * x))

        elif self.impl == "min_max":
            low, high = min_(x), max_(x)
            self.low.write(m * jnp.minimum(self.low.read(), low) + (1 - m) * low)
            self.high.write(m * jnp.maximum(self.high.read(), high) + (1 - m) * high)

        # DEFAULT
        elif self.impl == "perc_ema":
            low, high = per(x, self.perclo), per(x, self.perchi)
            self.low.write(m * self.low.read() + (1 - m) * low)
            self.high.write(m * self.high.read() + (1 - m) * high)

        elif self.impl == "perc_ema_corr":
            self.step.write(self.step.read() + 1)
            low, high = per(x, self.perclo), per(x, self.perchi)
            self.low.write(m * self.low.read() + (1 - m) * low)
            self.high.write(m * self.high.read() + (1 - m) * high)
        elif self.impl == "mean_mag":
            curr = mean(jnp.abs(x))
            self.mag.write(m * self.mag.read() + (1 - m) * curr)
        elif self.impl == "max_mag":
            curr = max_(jnp.abs(x))
            self.mag.write(m * jnp.maximum(self.mag.read(), curr) + (1 - m) * curr)
        else:
            raise NotImplementedError(self.impl)

    def stats(self):
        if self.impl == "off":
            return 0.0, 1.0

        elif self.impl == "mean_std":
            corr = 1 - self.decay ** self.step.read().astype(jnp.float32)
            mean = self.mean.read() / corr
            var = (self.sqrs.read() / corr) - self.mean.read() ** 2
            std = jnp.sqrt(jnp.maximum(var, 1 / self.max**2) + self.eps)
            return sg(mean), sg(std)

        elif self.impl == "min_max":
            offset = self.low.read()
            invscale = jnp.maximum(1 / self.max, self.high.read() - self.low.read())
            return sg(offset), sg(invscale)

        # DEFAULT
        elif self.impl == "perc_ema":
            offset = self.low.read()
            invscale = jnp.maximum(1 / self.max, self.high.read() - self.low.read())
            return sg(offset), sg(invscale)

        elif self.impl == "perc_ema_corr":
            corr = 1 - self.decay ** self.step.read().astype(jnp.float32)
            lo = self.low.read() / corr
            hi = self.high.read() / corr
            invscale = jnp.maximum(1 / self.max, hi - lo)
            return sg(lo), sg(invscale)

        elif self.impl == "mean_mag":
            offset = jnp.array(0)
            invscale = jnp.maximum(1 / self.max, self.mag.read())
            return sg(offset), sg(invscale)

        elif self.impl == "max_mag":
            offset = jnp.array(0)
            invscale = jnp.maximum(1 / self.max, self.mag.read())
            return sg(offset), sg(invscale)
        else:
            raise NotImplementedError(self.impl)


def late_grad_clip(value=1.0):
    def init_fn(params):
        return ()

    def update_fn(updates, state, params):
        updates = tree_map(lambda x: jnp.clip(x, -value, value), updates)
        return updates, ()

    return optax.GradientTransformation(init_fn, update_fn)


def tree_keys(params, prefix=""):
    if hasattr(params, "items"):
        return type(params)(
            {k: tree_keys(v, prefix + "/" + k.lstrip("/")) for k, v in params.items()}
        )
    elif isinstance(params, (tuple, list)):
        return [tree_keys(x, prefix) for x in params]
    elif isinstance(params, jnp.ndarray):
        return prefix
    else:
        raise TypeError(type(params))


class SlowUpdater:
    def __init__(self, src, dst, new_frac=1.0, period=1):
        self.src = src
        self.dst = dst  # slow
        self.new_frac = new_frac
        self.period = period
        self.steps = nj.Variable(jnp.zeros, (), jnp.int32, name="updates")

    def __call__(self):
        assert self.src.getm()
        steps = self.steps.read()
        need_init = (steps == 0).astype(jnp.float32)
        need_update = (steps % self.period == 0).astype(jnp.float32)
        new_mix = jnp.clip(1.0 * need_init + self.new_frac * need_update, 0, 1)
        source = {
            k.replace(f"/{self.src.name}/", f"/{self.dst.name}/"): v
            for k, v in self.src.getm().items()
        }
        self.dst.putm(
            tree_map(
                lambda s, d: new_mix * s + (1 - new_mix) * d, source, self.dst.getm()
            )
        )
        self.steps.write(steps + 1)


# class ModuleEMA:
#     """Exponential Moving Average of module parameters."""

#     def __init__(
#         self,
#         module: nj.Module,
#         decay=0.998,
#         update_every=1,
#         linear_warmup_steps=None,
#         initial_decay=None,
#     ):
#         """
#         Args:
#             module: The source module whose parameters to track with EMA
#             decay: EMA decay rate (higher = slower update)
#             update_every: Update EMA every N calls
#             name: Name for this EMA module
#             linear_warmup_steps: If provided, linearly increase decay over this many steps
#             initial_decay: Starting decay value for linear warmup (defaults to 0.0)
#         """
#         self.module = module
#         self.decay = decay
#         self.update_every = update_every
#         self.step_count = nj.Variable(jnp.zeros, (), jnp.int32, name="step_count")

#         # Linear warmup parameters
#         self.linear_warmup_steps = linear_warmup_steps
#         self.initial_decay = initial_decay if initial_decay is not None else 0.0
#         self.final_decay = decay

#         # Initialize EMA parameters - will be set on first call
#         self._ema_params = None
#         self._initialized = False

#     def _get_current_decay(self, step):
#         """Get current decay rate, potentially with linear warmup."""
#         if self.linear_warmup_steps is None:
#             return self.decay

#         # Linear interpolation from initial_decay to final_decay
#         progress = jnp.minimum(step / self.linear_warmup_steps, 1.0)
#         current_decay = self.initial_decay + progress * (
#             self.final_decay - self.initial_decay
#         )
#         return current_decay

#     def __call__(self, force_update=False):
#         """Update EMA parameters and return current EMA state.

#         Args:
#             force_update: Force update regardless of update_every setting

#         Returns:
#             Dictionary of EMA parameters
#         """
#         current_params = self.module.getm()
#         step = self.step_count.read()

#         # Initialize EMA parameters on first call
#         if not self._initialized:
#             self._ema_params = tree_map(lambda x: x.copy(), current_params)
#             self._initialized = True
#             return self._ema_params

#         # Check if we should update
#         should_update = force_update or (step % self.update_every == 0)

#         if should_update:
#             # Get current decay rate (potentially with linear warmup)
#             current_decay = self._get_current_decay(step)

#             # Apply EMA update: ema = decay * ema + (1 - decay) * current
#             self._ema_params = tree_map(
#                 lambda ema, current: current_decay * ema
#                 + (1 - current_decay) * current,
#                 self._ema_params,
#                 current_params,
#             )

#         self.step_count.write(step + 1)
#         return self._ema_params

#     def get_ema_params(self):
#         """Get current EMA parameters without updating."""
#         if not self._initialized:
#             return self.module.getm()
#         return self._ema_params

#     def reset_ema(self):
#         """Reset EMA to current module parameters."""
#         self._ema_params = tree_map(lambda x: x.copy(), self.module.getm())
#         self.step_count.write(0)


def symbolic_to_multihot(layers):
    # Concatenate entity and avatar layers
    # layers = jnp.concatenate((obs["entities"], obs["avatar"]), axis=-1).astype(int)
    layers = layers.astype(int)

    # Use advanced indexing to avoid explicit loops for one-hot encoding
    # This creates a one-hot encoding for each entity in each layer at once
    from ledwm.nets.EncoderSent import NUM_ALL_ENTITIES

    n_layers = layers.shape[-1]
    one_hot_encoded = jnp.eye(NUM_ALL_ENTITIES)[layers.reshape(-1)]
    one_hot_encoded = one_hot_encoded.reshape(
        *layers.shape[:-1], n_layers, NUM_ALL_ENTITIES
    )
    # Reduce across the layer dimension to get multi-hot encoding
    new_ob = jnp.max(one_hot_encoded, axis=-2)

    # new_ob: bs, bl, 10, 10, 17. set the first channel in the last dim to 0
    new_ob = new_ob.at[:, :, :, :, 0].set(0)

    # Assuming observation_space_shape is known and static, matching it is implicit
    # If you need to enforce the shape explicitly, consider reshaping or asserting outside this JAX function
    return new_ob


def mask_image_with_entity_mask(image, entity_mask, entity_pos):
    # We need to update only where entity_mask is 0
    bs, Ne, _ = entity_pos.shape
    inactive_indices = jnp.where(entity_mask.reshape(bs * Ne) == 0)

    # Flattening the positions since we have converted to a linear index
    x_positions = entity_pos[:, :, 0].reshape(bs * Ne)[inactive_indices]
    y_positions = entity_pos[:, :, 1].reshape(bs * Ne)[inactive_indices]
    z_positions = entity_pos[:, :, 2].reshape(bs * Ne)[inactive_indices]
    # Creating batch indices for each entry
    batch_indices = jnp.repeat(jnp.arange(bs), Ne)[inactive_indices]

    # Update the image at the specified positions
    image = image.at[batch_indices, x_positions, y_positions, z_positions].set(0)
    return image


def test_mask():
    bs = 1  # Batch size
    Ne = 3  # Number of entities
    height, width, channels = 10, 10, 17  # Image dimensions

    # Define the positions of entities and their mask
    entity_mask = jnp.array([[0, 1, 0]])  # Only the second entity is active
    entity_pos = jnp.array(
        [
            [
                [3, 5, 1],  # Position of the first entity
                [7, 5, 3],  # Position of the second entity
                [5, 7, 10],  # Position of the third entity
            ]
        ]
    )

    # Initialize the image with zeros
    image = jnp.zeros((bs, height, width, channels))
    for pos in entity_pos[0]:
        x, y, z = pos  # Extract x, y, z coordinates from each position
        # print(x, y, z)
        image = image.at[0, x, y, z].set(1)  # Set the value at the specified position
    # print nonzeros of image
    print(jnp.nonzero(image))
    new_image = mask_image_with_entity_mask(image, entity_mask, entity_pos)
    print(jnp.nonzero(new_image))


def fill_positions(image, pos):
    """
    pos: (Ne, 3)
    """
    x, y, depth_index = pos.T  # (Ne, 3) -> (3, Ne)
    image = image.at[x, y, depth_index].set(1)
    return image


def create_batch_images_from_pos(entity_pos, avatar_pos, W, H, D=17):
    """
    entity_pos: (bs, Ne, 3)
    avatar_pos: (bs, 1, 3)
    """
    assert entity_pos.ndim == avatar_pos.ndim, (entity_pos.ndim, avatar_pos.ndim)
    has_bs = True
    if entity_pos.ndim == 2:  # only has 1 sample
        entity_pos = entity_pos[None, :]
        has_bs = False

    if avatar_pos.ndim == 2:
        avatar_pos = avatar_pos[None, :]

    # bs is the other dim except the last 2 dims: (Ne, 3) of entity_pos
    # print(f"{entity_pos.shape=}")
    batch_dims = entity_pos.shape[:-2]  #
    entity_pos = entity_pos.reshape(-1, *entity_pos.shape[-2:])
    avatar_pos = avatar_pos.reshape(-1, *avatar_pos.shape[-2:])

    bs = entity_pos.shape[0]
    images = jnp.zeros((bs, W, H, D))
    # take each entity_pos and avatar_pos, and fill the images
    fill_positions_batched = jax.vmap(fill_positions, in_axes=(0, 0), out_axes=0)
    images = fill_positions_batched(images, entity_pos)
    images = fill_positions_batched(images, avatar_pos)
    images = images.reshape(batch_dims + (W, H, D))

    if not has_bs:
        images = images[0]

    return images


def mean_over_nonzero(grid):
    # grid: bs, 16, 16, 17, d_entity
    # sum over dim 17, then divide by Ne + 1
    # grid_mean = jnp.sum(grid, axis=3) / (Ne + 1)
    # count the number of nonzero vectors in dim=3 (17)
    nonzero = jnp.count_nonzero(grid, axis=3)
    nonzero = jnp.where(nonzero == 0, 1, nonzero)
    return jnp.sum(grid, axis=3) / nonzero


def put_atten_to_grid(
    entity_pos, entity_emb, avatar_pos, avatar_embed, grid_size=(16, 16)
):
    """
    Args:
        entity_pos: (bs, Ne, 3) the last dimension is for (16, 16, 17)
        entity_emb: (bs, Ne, d_entity) the last dimension is for embeddings
        avatar_pos: (bs, 1, 3) the last dimension is for (16, 16, 17)
        avatar_embed: (bs, 1, d_avatar) the last dimension is for embeddings
    Returns:
        grid_mean: (bs, 16, 16, d_entity)
    """
    bs, Ne, _ = entity_pos.shape
    _, _, d_entity = entity_emb.shape
    _, _, d_avatar = avatar_embed.shape
    assert d_entity == d_avatar, f"d_entity: {d_entity}, d_avatar: {d_avatar}"

    entity_emb, avatar_embed = cast_to_compute((entity_emb, avatar_embed))

    # Accumulate embeddings directly into the spatial grid. The previous
    # representation materialized a (W, H, 17, D) tensor only to immediately
    # reduce the 17 position slots again. Keep a small occupancy grid to retain
    # the same per-position averaging semantics without the 17x feature buffer.
    grid = jnp.zeros((bs, *grid_size, d_entity), entity_emb.dtype)
    occupied = jnp.zeros((bs, *grid_size, 17), jnp.int32)
    entity_x_coords = entity_pos[:, :, 0].astype(jnp.int32)
    entity_y_coords = entity_pos[:, :, 1].astype(jnp.int32)
    entity_z_coords = entity_pos[:, :, 2].astype(jnp.int32)
    entity_valid = (
        (entity_x_coords >= 0)
        & (entity_x_coords < grid_size[0])
        & (entity_y_coords >= 0)
        & (entity_y_coords < grid_size[1])
        & (entity_z_coords >= 0)
        & (entity_z_coords < 17)
    )
    entity_x_coords = jnp.where(entity_valid, entity_x_coords, 0)
    entity_y_coords = jnp.where(entity_valid, entity_y_coords, 0)
    entity_z_coords = jnp.where(entity_valid, entity_z_coords, 0)
    b_idx, _ = jnp.meshgrid(jnp.arange(bs), jnp.arange(Ne), indexing="ij")
    entity_index = (
        b_idx.ravel(),  # first dim of all Ne
        entity_x_coords.ravel(),
        entity_y_coords.ravel(),
        slice(None),
    )
    entity_emb = rearrange(
        entity_emb * entity_valid[..., None], "bs Ne d -> (bs Ne) d"
    )
    grid = grid.at[entity_index].add(entity_emb)
    occupied = occupied.at[
        (
            b_idx.ravel(),
            entity_x_coords.ravel(),
            entity_y_coords.ravel(),
            entity_z_coords.ravel(),
        )
    ].add(entity_valid.ravel().astype(jnp.int32))

    avatar_x_coords = avatar_pos[:, :, 0].astype(jnp.int32)
    avatar_y_coords = avatar_pos[:, :, 1].astype(jnp.int32)
    avatar_z_coords = avatar_pos[:, :, 2].astype(jnp.int32)
    avatar_valid = (
        (avatar_x_coords >= 0)
        & (avatar_x_coords < grid_size[0])
        & (avatar_y_coords >= 0)
        & (avatar_y_coords < grid_size[1])
        & (avatar_z_coords >= 0)
        & (avatar_z_coords < 17)
    )
    avatar_x_coords = jnp.where(avatar_valid, avatar_x_coords, 0)
    avatar_y_coords = jnp.where(avatar_valid, avatar_y_coords, 0)
    avatar_z_coords = jnp.where(avatar_valid, avatar_z_coords, 0)
    avatar_index = (
        jnp.arange(bs),
        avatar_x_coords.ravel(),
        avatar_y_coords.ravel(),
        slice(None),
    )
    avatar_emb = rearrange(
        avatar_embed * avatar_valid[..., None], "bs 1 d -> bs d"
    )
    grid = grid.at[avatar_index].add(avatar_emb)
    occupied = occupied.at[
        (
            jnp.arange(bs),
            avatar_x_coords.ravel(),
            avatar_y_coords.ravel(),
            avatar_z_coords.ravel(),
        )
    ].add(avatar_valid.ravel().astype(jnp.int32))

    count = jnp.maximum(jnp.count_nonzero(occupied, axis=3), 1)[..., None]
    grid_mean = grid / count.astype(grid.dtype)
    assert grid_mean.shape == (
        bs,
        *grid_size,
        d_entity,
    ), f"grid_mean.shape: {grid_mean.shape}"
    return grid_mean


def put_atten_to_grid_seperate(
    entity_pos, entity_emb, avatar_pos, avatar_embed, grid_size=(16, 16)
):
    """
    Args:
        entity_pos: (bs, Ne, 3) the last dimension is for (16, 16, 17)
        entity_emb: (bs, Ne, d_entity) the last dimension is for embeddings
        avatar_pos: (bs, 1, 3) the last dimension is for (16, 16, 17)
        avatar_embed: (bs, 1, d_avatar) the last dimension is for embeddings
    Returns:
        grid_mean: (bs, Ne, 16, 16, d_entity)
        Each grid (bs, i) contains entity_emb[i] and avatar_embed
    """
    bs, Ne, _ = entity_pos.shape
    _, _, d_entity = entity_emb.shape
    _, _, d_avatar = avatar_embed.shape
    assert d_entity == d_avatar, f"d_entity: {d_entity}, d_avatar: {d_avatar}"

    entity_emb, avatar_embed = cast_to_compute((entity_emb, avatar_embed))

    # As above, avoid materializing the position-slot axis together with every
    # feature channel. Only occupancy needs the 17 slots.
    grid = jnp.zeros((bs, Ne, *grid_size, d_entity), entity_emb.dtype)
    occupied = jnp.zeros((bs, Ne, *grid_size, 17), jnp.int32)

    # For each entity, create separate grids
    entity_x_coords = entity_pos[:, :, 0].astype(jnp.int32)
    entity_y_coords = entity_pos[:, :, 1].astype(jnp.int32)
    entity_z_coords = entity_pos[:, :, 2].astype(jnp.int32)
    entity_valid = (
        (entity_x_coords >= 0)
        & (entity_x_coords < grid_size[0])
        & (entity_y_coords >= 0)
        & (entity_y_coords < grid_size[1])
        & (entity_z_coords >= 0)
        & (entity_z_coords < 17)
    )
    entity_x_coords = jnp.where(entity_valid, entity_x_coords, 0)
    entity_y_coords = jnp.where(entity_valid, entity_y_coords, 0)
    entity_z_coords = jnp.where(entity_valid, entity_z_coords, 0)

    # Create indices for each entity in its own grid
    b_idx, e_idx = jnp.meshgrid(jnp.arange(bs), jnp.arange(Ne), indexing="ij")
    entity_index = (
        b_idx.ravel(),  # batch dimension
        e_idx.ravel(),  # entity dimension
        entity_x_coords.ravel(),
        entity_y_coords.ravel(),
        slice(None),
    )
    entity_emb_flat = rearrange(
        entity_emb * entity_valid[..., None], "bs Ne d -> (bs Ne) d"
    )
    grid = grid.at[entity_index].add(entity_emb_flat)
    occupied = occupied.at[
        (
            b_idx.ravel(),
            e_idx.ravel(),
            entity_x_coords.ravel(),
            entity_y_coords.ravel(),
            entity_z_coords.ravel(),
        )
    ].add(entity_valid.ravel().astype(jnp.int32))

    # Add avatar to each entity's grid
    avatar_x_coords = avatar_pos[:, :, 0].astype(jnp.int32)
    avatar_y_coords = avatar_pos[:, :, 1].astype(jnp.int32)
    avatar_z_coords = avatar_pos[:, :, 2].astype(jnp.int32)
    avatar_valid = (
        (avatar_x_coords >= 0)
        & (avatar_x_coords < grid_size[0])
        & (avatar_y_coords >= 0)
        & (avatar_y_coords < grid_size[1])
        & (avatar_z_coords >= 0)
        & (avatar_z_coords < 17)
    )
    avatar_x_coords = jnp.where(avatar_valid, avatar_x_coords, 0)
    avatar_y_coords = jnp.where(avatar_valid, avatar_y_coords, 0)
    avatar_z_coords = jnp.where(avatar_valid, avatar_z_coords, 0)

    # Broadcast avatar to all entity grids
    avatar_index = (
        jnp.arange(bs)[:, None],  # batch dimension
        jnp.arange(Ne)[None, :],  # entity dimension (broadcast to all entities)
        avatar_x_coords[:, 0, None],  # x coordinate
        avatar_y_coords[:, 0, None],  # y coordinate
        slice(None),
    )
    # (bs, 1, d) -> (bs, Ne, d)
    avatar_emb_broadcast = repeat(
        avatar_embed * avatar_valid[..., None], "bs 1 d -> bs Ne d", Ne=Ne
    )
    grid = grid.at[avatar_index].add(avatar_emb_broadcast)
    occupied = occupied.at[
        (
            jnp.arange(bs)[:, None],
            jnp.arange(Ne)[None, :],
            avatar_x_coords[:, 0, None],
            avatar_y_coords[:, 0, None],
            avatar_z_coords[:, 0, None],
        )
    ].add(
        jnp.broadcast_to(avatar_valid[:, 0, None], (bs, Ne)).astype(jnp.int32)
    )

    count = jnp.maximum(jnp.count_nonzero(occupied, axis=4), 1)[..., None]
    grid_mean = grid / count.astype(grid.dtype)
    assert grid_mean.shape == (
        bs,
        Ne,
        *grid_size,
        d_entity,
    ), f"grid_mean.shape: {grid_mean.shape}"
    return grid_mean


def dropout2d(x, drop_prob, rng_key, training=True):
    """Applies Dropout to the input array x at the specified dropout probability.

    Args:
        x (jax.numpy.ndarray): Input data (batch_size, height, width, channels).
        drop_prob (float): Probability of dropping a unit (channel).
        rng_key (jax.random.PRNGKey): Random key.
        is_training (bool): Whether the dropout is applied or the original array is returned.

    Returns:
        jax.numpy.ndarray: Array with the same shape as x with dropout applied.
    """
    if not training or drop_prob == 0:
        return x
    else:
        assert len(x.shape) == 4, x.shape
        # Create a random mask for the channels
        keep_prob = 1 - drop_prob
        # Generate a random array with the shape of the channel dimension
        random_mask = random.bernoulli(rng_key, p=keep_prob, shape=(x.shape[-1],))
        # Broadcast the mask to the shape of x and scale by keep_prob for expected value
        mask = jnp.expand_dims(
            random_mask, axis=(0, 1, 2)
        )  # Expand dimensions to broadcast
        return x * mask / keep_prob


def test_dropout2d():
    # Create a random PRNG key
    key = random.PRNGKey(0)

    # Simulate some input data (e.g., a batch of images with shape (batch, height, width, channels))
    x = random.normal(key, (10, 32, 32, 3))  # 10 images, 32x32 pixels, 3 channels

    # Apply 2D dropout
    drop_prob = 0.5  # 50% probability to drop each channel
    is_training = True  # Apply dropout only during training
    dropped_x = dropout2d(x, drop_prob, key, is_training)

    print(dropped_x.shape)
    print(dropped_x[0, :, :, 2])
    print(dropped_x[0, :, :, 1])
    print(dropped_x[0, :, :, 0])


import os
import jax.numpy as jnp


def apply_loss_mask(is_last, loss):
    bs, bl = is_last.shape
    # Convert is_last from 0/1 to boolean
    is_last_bool = is_last.astype(bool)

    # Find the index of the first 'True' in each batch
    first_true_index = jnp.argmax(is_last_bool, axis=1)

    # If 'True' is never found, it returns 0, which could be misleading if the first element isn't True.
    # We need to verify and adjust this case:
    first_true_index = jnp.where(jnp.any(is_last_bool, axis=1), first_true_index, bl)

    # Generate a mask where each position is less than or equal to the first 'True' index
    mask = jnp.arange(bl)[:, None] <= first_true_index

    # Apply the mask, setting losses after 'True' to zero
    masked_loss = jnp.where(mask.T, loss, 0)
    return masked_loss


def get_task(config):
    return config.task.split("_")[1]


def create_horizon_mask_from(
    is_first,
    is_last,
    horizon,
    include_current_step=False,
):
    """
    returns: bs, bl, K
    """
    assert is_first.ndim == 2, is_first.shape
    bs, bl = is_first.shape
    MAX = bl + 1

    def process_single_batch(is_first, is_last):
        end_indices = jnp.where(is_last, jnp.arange(bl), MAX)
        end_indices = lax.cummin(end_indices[::-1])[::-1]
        end_indices = jnp.where(end_indices == MAX, bl - 1, end_indices)
        # print(f"{end_indices=}")
        if include_current_step:
            future_range = jnp.arange(horizon)
        else:
            future_range = jnp.arange(1, horizon + 1)
        time_steps = future_range + jnp.arange(bl)[:, None]
        end_indices = end_indices[:, None]
        mask = (time_steps <= end_indices).astype(jnp.int32)
        return mask

    horizon_masks = jax.vmap(process_single_batch)(is_first, is_last)
    return horizon_masks


def extract_horizon_data_from(
    data,  # bs, bl, *d
    horizon,  # int
    mask=None,  # bs, bl, K
    is_first=None,  # bs, bl
    is_last=None,  # bs, bl
    include_current_step=False,  # horizon includes current step,
    mask_value_id=None,  # for one hot - set mask_value_id=1
):
    """ "
    returns: bs, bl, K, *d
    from each step, extract the next h steps including the current step
    """
    bs, bl, *d = data.shape

    # Create the horizon mask
    if mask is None:
        assert is_first is not None and is_last is not None
        mask = create_horizon_mask_from(
            is_first, is_last, horizon, include_current_step
        )  # bs, bl, K

    # pad to data bl + K
    data = jnp.concatenate([data, jnp.zeros((bs, horizon, *d))], axis=1)

    def extract_one_batch(data):
        def extract_one_step(i):
            # data now is bl, *d
            # print(f'{data=}')
            if include_current_step:
                next_index = i
            else:
                next_index = i + 1
            return lax.dynamic_slice(
                data, [next_index] + [0] * len(data.shape[1:]), [horizon, *d]
            )

        return jax.vmap(extract_one_step)(jnp.arange(bl))  # bl, h, *d

    res = jax.vmap(extract_one_batch)(data)  # bs, bl, h, *d
    mask = mask.reshape(bs, bl, horizon, *([1] * len(d)))  # bs, bl, h, 1

    if mask_value_id is not None:
        mask = rearrange(mask, "... 1 -> ...")
        res = res.at[..., mask_value_id].set(
            jnp.where(mask == 0, 1, res[..., mask_value_id])
        )
        return res
    else:
        # print(f"{mask=}")
        return res * mask


def multi_hot(labels, num_classes: int):
    # Get one-hot encoded vectors for each label
    one_hot_vectors = jnp.eye(num_classes)[labels]

    # Sum across the rows to get a multi-hot vector (combining all one-hot vectors)
    multi_hot_vector = one_hot_vectors.sum(axis=0)

    # Ensure the output is boolean (either 0 or 1 per class, not sums)
    multi_hot_vector = jnp.clip(multi_hot_vector, 0, 1)

    return multi_hot_vector


def keep_is_first_only(context, max_env_len):
    bs, bl = context["is_first"].shape[:2]
    start = tree_map(lambda x: x.reshape((-1, *x.shape[2:])), context)
    min_num_eps = int(bs * (bl / max_env_len))

    # import inspect
    # print(inspect.getfile(jnp.extract))
    def extract(is_first, x):
        if x.shape[0] != is_first.shape[0]:
            return x
        # is_first: bs*bl -> broadcast to the shape of x
        is_first = jnp.broadcast_to(
            is_first.reshape(-1, *((1,) * (x.ndim - 1))), x.shape
        )
        assert is_first.shape == x.shape, (is_first.shape, x.shape)
        extract_shape = (min_num_eps,) + x.shape[1:]
        # size = product of all dims
        return jnp.extract(
            is_first, x, size=np.prod(extract_shape), fill_value=0
        ).reshape(extract_shape)
        # min_num_eps, d*

    start = tree_map(lambda x: extract(start["is_first"], x), start)
    mask = start["is_first"] != 0
    # jax.debug.print("start['is_first'] = {x}", x=start)
    jax.debug.print("mask = {x}", x=mask)
    return start, mask


def draw_cont_prob_hist(report):
    # return fig
    in_mask = report["rollout_cont_mask"] == 1
    probs = report["rollout_cont_prob"][in_mask]
    fig = plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=50, color="blue", alpha=0.7)
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Continuity Probabilities")
    plt.grid()
    return fig


def draw_precision_recall_curve(report):
    in_mask = report["rollout_cont_mask"] == 1
    y_prob = report["rollout_cont_prob"][in_mask]
    y_test = report["rollout_cont_label"][in_mask]

    # Compute precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    # Extend thresholds to include edge case (all negatives classified)
    # Add 0, 1 as the final threshold for consistency
    thresholds = np.append(thresholds, 1)

    # Calculate F1 scores for each threshold
    f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]

    # Find the best threshold for maximum F1 score
    # assert f1_scores is numpy array
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]

    # Plot Precision-Recall curve
    fig = plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
    plt.scatter(
        recall[best_threshold_idx],
        precision[best_threshold_idx],
        color="red",
        label=f"Best Threshold ({best_threshold:.2f})",
        zorder=5,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve with Best Threshold")
    plt.legend()
    plt.grid()
    return fig


def test_horizon_data():
    is_first = jnp.array([[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]])
    is_last = jnp.array([[0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 0, 1]])
    data = jnp.array(
        [
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
            [[6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
        ]
    )  # bs=2, bl=5, d=2
    horizon_mask_from_next = create_horizon_mask_from(is_first, is_last, 2)
    print(f"{horizon_mask_from_next=}")
    horizon_data = extract_horizon_data_from(
        data, horizon=2, is_first=is_first, is_last=is_last, mask=horizon_mask_from_next
    )
    print(horizon_data.shape)
    print(horizon_data)


def onehot_to_float(tgt, reward_values):
    return reward_values[jnp.argmax(tgt, axis=-1)]


def test_mask_value_id():
    import jax
    import jax.numpy as jnp

    # Assume `res` has shape (bs, bl, K, d)
    # Assume `mask` has shape (bs, bl, K, 1) (as per the reshape)
    # `mask_value_id` is the index to set to 1 when mask == 0

    # Example shapes
    bs, bl, h, d = 2, 3, 4, 5  # Batch size, sequence length, horizon, depth
    res = jnp.zeros((bs, bl, h, d))  # Example tensor
    mask = jnp.array(
        [
            [[[1], [0], [1], [0]], [[1], [1], [0], [0]], [[0], [1], [1], [1]]],
            [[[1], [0], [1], [1]], [[0], [1], [1], [0]], [[1], [1], [0], [1]]],
        ]
    )  # Shape (bs, bl, h, 1)
    mask_value_id = 2  # Example mask_value_id

    # Squeeze mask to match the shape needed
    mask = mask.squeeze(-1)  # Shape (bs, bl, h)

    # Use `.at` to update `res` at the specified index where `mask == 0`
    res = res.at[..., mask_value_id].set(
        jnp.where(mask == 0, 1, res[..., mask_value_id])
    )

    print("Updated res:")
    print(res)


def check_nan(tensor, raise_error=True):
    """
    Check if a JAX tensor contains NaN values. For traced tensors, this function
    attempts to evaluate the tensor or perform a symbolic check.

    Args:
        tensor: JAX tensor to check for NaN values.
        raise_error: If True, raises an AssertionError if NaNs are found.
                     If False, returns a boolean indicating if NaNs are present.

    Returns:
        bool: True if NaNs are present, False otherwise (if raise_error=False).
    """
    try:
        # Attempt to evaluate the tensor if it's not traced
        tensor_np = jax.device_get(tensor)
        has_nan = np.any(np.isnan(tensor_np))
        if raise_error and has_nan:
            raise AssertionError("Tensor contains NaN values")
        return has_nan
    except Exception:
        # If evaluation fails (e.g., tensor is traced), perform symbolic check
        has_nan = jnp.any(jnp.isnan(tensor))
        if raise_error:
            # Note: Assertions inside JIT contexts are not supported,
            # so this might need to be handled differently in such cases
            raise AssertionError("Tensor contains NaN values (symbolic check)")
        return has_nan


def test_create_batch_images_from_pos():
    entity_pos = jnp.array([[[0, 0, 2], [15, 15, 4]]])
    avatar_pos = jnp.array([[[1, 1, 15], [8, 8, 16]]])
    images_batch = create_batch_images_from_pos(entity_pos, avatar_pos, 16, 16, 17)
    print(images_batch.shape)
    # print nonzero
    print(jnp.count_nonzero(images_batch))
    print(jnp.nonzero(images_batch))


def test_put_atten_to_grid_seperate():
    """Test the put_atten_to_grid_seperate function"""
    bs, Ne, d_entity = 2, 3, 64

    # Create test data
    entity_pos = jnp.array(
        [
            [[1, 2, 5], [3, 4, 7], [5, 6, 9]],  # batch 0
            [[2, 3, 6], [4, 5, 8], [6, 7, 10]],  # batch 1
        ]
    )  # (bs=2, Ne=3, 3)

    entity_emb = (
        jnp.ones((bs, Ne, d_entity)) * jnp.arange(Ne)[None, :, None]
    )  # Different values per entity
    avatar_pos = jnp.array([[[0, 0, 0]], [[1, 1, 1]]])  # (bs=2, 1, 3)
    avatar_embed = jnp.ones((bs, 1, d_entity)) * 100  # (bs=2, 1, d_entity)

    # Test the function
    result = put_atten_to_grid_seperate(
        entity_pos, entity_emb, avatar_pos, avatar_embed
    )

    print(f"Input shapes:")
    print(f"  entity_pos: {entity_pos.shape}")
    print(f"  entity_emb: {entity_emb.shape}")
    print(f"  avatar_pos: {avatar_pos.shape}")
    print(f"  avatar_embed: {avatar_embed.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: (bs={bs}, Ne={Ne}, 16, 16, d_entity={d_entity})")

    # Verify each entity grid has its own embedding + avatar
    for b in range(bs):
        for e in range(Ne):
            entity_x, entity_y, entity_z = entity_pos[b, e]
            avatar_x, avatar_y, avatar_z = avatar_pos[b, 0]

            # Check that entity position has entity embedding
            entity_val = result[
                b, e, entity_x, entity_y, 0
            ]  # First dimension of embedding
            print(
                f"Batch {b}, Entity {e}: entity_val at pos ({entity_x},{entity_y}) = {entity_val}"
            )

            # Check that avatar position has avatar embedding
            avatar_val = result[
                b, e, avatar_x, avatar_y, 0
            ]  # First dimension of embedding
            print(
                f"Batch {b}, Entity {e}: avatar_val at pos ({avatar_x},{avatar_y}) = {avatar_val}"
            )


def add_dummy_first_action(action):
    return jnp.concatenate([jnp.zeros_like(action[:, :1]), action[:, :-1]], axis=1)


def sample_z_from(dist, argmax=False):
    if argmax:
        if hasattr(dist, "argmax_onehot"):
            return dist.argmax_onehot()
        if hasattr(dist, "distribution") and hasattr(
            dist.distribution, "argmax_onehot"
        ):
            return dist.distribution.argmax_onehot()
        raise AttributeError(f"{type(dist).__name__} does not support argmax_onehot")
    else:
        return dist.sample(seed=nj.rng())


def get_dist_z_from(stats, impl, step=None, soft_z=False):
    f32 = jnp.float32
    from ledwm.nets.Dist import OneHotDist
    import ledwm.tfp_compat  # noqa: F401
    from tensorflow_probability.substrates.jax import distributions as tfd

    if impl == "gaussian":
        raise NotImplementedError

    elif impl == "softmax":
        logit = stats["logit"].astype(f32)

        if soft_z:
            raise NotImplementedError
        else:
            return tfd.Independent(OneHotDist(logit), 1)

    elif impl == "maskgit":
        # logit = stats["logit"].astype(f32)
        # return jaxutils.OneHotDist(logit)
        raise NotImplementedError

    else:
        raise NotImplementedError


if __name__ == "__main__":
    # test_horizon_data()
    # test_ema()
    test_put_atten_to_grid_seperate()
