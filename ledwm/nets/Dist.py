from ledwm import jaxutils, ninjax as nj
from ledwm.nets.Linear import LinearAct
from jax import lax
from jax.core import Tracer


import jax.numpy as jnp
import ledwm.tfp_compat  # noqa: F401
from tensorflow_probability.substrates.jax import distributions as tfd
from ledwm.jaxutils import sg, symexp, symlog
from ledwm.nets import f32, tfd

import jax
import jax.numpy as jnp
import numpy as np

import ledwm.tfp_compat  # noqa: F401
from tensorflow_probability.substrates import jax as tfp


def find_class_weight(img_logits, task):  # bs, bl, w, h, 17
    total_pixels = np.prod(img_logits.shape[-3:])

    from ledwm.embodied.envs.MessengerSent import NUM_SENTS

    assert task is not None
    num_ones = NUM_SENTS[task] + 1  # +1 agent
    class_weights = [1, 2]
    print(f"distribution.config | type=weighted_bernoulli | weights={class_weights}")
    return class_weights


class StableBernoulli(tfd.Bernoulli):
    def __init__(self, logits=None, probs=None, **kwargs):
        super(StableBernoulli, self).__init__(logits=logits, probs=probs, **kwargs)

    def log_prob(self, value, name="log_prob"):
        """Compute the log probabilities in a numerically stable way."""
        # Ensure value is compatible with logits' shape
        value = jnp.broadcast_to(value, self.logits_parameter().shape)

        # Use logits directly for a numerically stable calculation of log_prob
        # This avoids the numerical instability that comes from calculating probabilities first
        logits = self.logits_parameter()
        return -jnp.logaddexp(0.0, jnp.where(value == 1, -logits, logits))


def log_sigmoid_jax(x):
    # Direct computation using the definition of log sigmoid
    return -jnp.logaddexp(0, -x)


class FocalLossBernoulli(tfd.Bernoulli):
    def __init__(self, logits, alpha=0.25, gamma=2, **kwargs):
        super(FocalLossBernoulli, self).__init__(logits=logits, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        parameters = dict(locals())
        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        params = super()._parameter_properties(dtype=dtype)
        params["alpha"] = tfp.util.ParameterProperties()
        params["gamma"] = tfp.util.ParameterProperties()
        return params

    def log_prob(self, value, name="log_prob"):
        # Compute stable log_sigmoid for both logits and its negation
        log_sigmoid = log_sigmoid_jax(self.logits)
        log_sigmoid_neg = log_sigmoid_jax(-self.logits)

        # Compute log(p_t) in a stable way
        log_p_t = jnp.where(value == 1, log_sigmoid, log_sigmoid_neg)
        # Compute p_t from its log form if needed for other computations
        p_t = jnp.exp(log_p_t)

        # Determine alpha_t based on the value (class)
        alpha_t = jnp.where(value == 1, 1 - self.alpha, self.alpha)  # 0.75, 0.25

        # Compute the focal modulation in the log space for numerical stability
        # Adding a small epsilon to avoid log(0)
        focal_modulation = self.gamma * jnp.log(1 - p_t + 1e-9)

        # Combine all components for the final focal loss, using log properties to stay in log space
        focal_loss = -alpha_t * jnp.exp(focal_modulation + log_p_t)

        return focal_loss


class WeightedBernoulli(tfd.Bernoulli):
    def __init__(self, logits=None, task=None, **kwargs):
        super(WeightedBernoulli, self).__init__(logits=logits, **kwargs)

        self.class_weights = jnp.array(
            find_class_weight(logits, task), dtype=self.dtype
        )
        parameters = dict(locals())
        self._parameters = parameters

    def log_prob(self, value, name="log_prob"):
        log_probs = super(WeightedBernoulli, self).log_prob(value, name=name)
        weights = jnp.where(value == 1, self.class_weights[1], self.class_weights[0])
        weighted_log_probs = log_probs * weights
        return weighted_log_probs

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        params = super()._parameter_properties(dtype=dtype)
        params["class_weights"] = tfp.util.ParameterProperties()
        return params


class OneHotDist(tfd.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype=jnp.float32):
        super().__init__(logits, probs, dtype)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return super()._parameter_properties(dtype)

    def sample(self, sample_shape=(), seed=None):
        sample = sg(super().sample(sample_shape, seed))
        probs = self._pad(super().probs_parameter(), sample.shape)
        return sg(sample) + (probs - sg(probs)).astype(sample.dtype)  # straight-through

    def _pad(self, tensor, shape):
        while len(tensor.shape) < len(shape):
            tensor = tensor[None]
        return tensor

    # attribute
    @property
    def probs(self):
        if self._probs is not None:
            return self._probs
        else:
            return jax.nn.softmax(self.logits)

    # Add the argmax method
    def argmax(self):
        """
        Returns the index of the highest probability category (argmax) from the distribution.
        """
        return jnp.argmax(self.probs, axis=-1)

    def mode(self, values=None):
        if values is not None:
            return values[self.argmax()]
        else:
            return super().mode()

    def argmax_onehot(self, sample_shape=(), seed=None):
        probs = self._pad(super().probs_parameter(), sample_shape)
        argmax_indices = jnp.argmax(probs, axis=-1)
        one_hot = jax.nn.one_hot(argmax_indices, num_classes=probs.shape[-1])
        # straight-through
        return sg(one_hot) + (probs - sg(probs)).astype(one_hot.dtype)


class MSEDist:
    def __init__(self, mode, dims=None, agg="sum"):
        self._mode = mode
        self._agg = agg
        self._dims = dims
        if dims is not None:
            self._dims = tuple([-x for x in range(1, dims + 1)])
            self.batch_shape = mode.shape[: len(mode.shape) - dims]
            self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            if self._dims:
                loss = distance.mean(self._dims)
            else:
                loss = distance
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            print("distribution.config | aggregation=none")
        #     raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dims, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)

        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = jnp.where(distance < self._tol, 0, distance)

        elif self._dist == "abs":
            distance = jnp.abs(self._mode - symlog(value))
            distance = jnp.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)

        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class TwoHotDist:
    def __init__(self, logits, bins, dims=0, transfwd=None, transbwd=None):
        assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
        self.logits = logits
        self.probs = jax.nn.softmax(logits)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = jnp.array(bins)
        self.transfwd = transfwd or (lambda x: x)  # symlog
        self.transbwd = transbwd or (lambda x: x)  # symexp
        self.batch_shape = logits.shape[: len(logits.shape) - dims - 1]
        self.event_shape = logits.shape[len(logits.shape) - dims : -1]

    def mean(self):
        return self.transbwd((self.probs * self.bins).sum(-1))

    def mode(self):
        return self.transbwd((self.probs * self.bins).sum(-1))

    def target(self, x):
        x = self.transfwd(x)  # symlog
        below = (self.bins <= x[..., None]).astype(jnp.int32).sum(-1) - 1
        above = len(self.bins) - (self.bins > x[..., None]).astype(jnp.int32).sum(-1)

        below = jnp.clip(below, 0, len(self.bins) - 1)
        above = jnp.clip(above, 0, len(self.bins) - 1)
        equal = below == above

        dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - x))
        dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - x))

        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        target = (
            jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None]
            + jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        )
        return target

    def log_pred(self):
        return self.logits - jax.scipy.special.logsumexp(self.logits, -1, keepdims=True)

    def log_prob(self, x, mask=None):
        res = self.target(x) * self.log_pred()
        return res.sum(-1).sum(self.dims)

    def kl(self, x):
        return kl_div(self.target(x), self.probs).sum(-1).sum(self.dims)


class Dist(nj.Module):
    def __init__(
        self,
        shape,
        dist="mse",
        outscale=0.1,
        outnorm=False,
        minstd=1.0,
        maxstd=1.0,
        unimix=0.0,
        bins=255,
        bound=20,
        unimix_decay="none",
    ):
        assert all(isinstance(dim, int) for dim in shape), shape
        self._shape = shape
        self._dist = dist
        self._minstd = minstd
        self._maxstd = maxstd
        self._unimix = unimix
        self._unimix_decay = unimix_decay
        self._outscale = outscale
        self._outnorm = outnorm
        self._bins = bins
        self._bound = bound
        assert self._bound > 0, f"bound must be positive, got {self._bound}"

    def __call__(self, inputs, step=None):
        dist = self.inner(inputs, step)
        assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
            dist.batch_shape,
            dist.event_shape,
            inputs.shape,
        )
        return dist

    def inner(self, inputs, step=None):
        kw = {}
        kw["outscale"] = self._outscale
        kw["outnorm"] = self._outnorm
        shape = self._shape
        if self._dist.endswith("_twohot"):
            shape = (*self._shape, self._bins)

        out = self.get("out", LinearAct, int(np.prod(shape)), **kw)(inputs)
        out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
        std = None

        if self._dist in ("normal", "trunc_normal"):
            std = self.get("std", LinearAct, int(np.prod(self._shape)), **kw)(inputs)
            std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
            # raise NotImplementedError

        if self._dist == "symlog_mse":
            return SymlogDist(out, len(self._shape), "mse", "sum")

        if self._dist == "symlog_and_twohot":
            bins = np.linspace(-self._bound, self._bound, out.shape[-1])
            return TwoHotDist(
                out, bins, len(self._shape), jaxutils.symlog, jaxutils.symexp
            )
            # raise NotImplementedError

        if self._dist == "symexp_twohot":
            bins = jaxutils.symexp(
                np.linspace(-self._bound, self._bound, out.shape[-1])
            )
            return TwoHotDist(out, bins, len(self._shape))

        if self._dist == "parab_twohot":
            eps = 0.001
            f = lambda x: np.sign(x) * (
                np.square(
                    np.sqrt(1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps
                )
                - 1
            )
            bins = f(np.linspace(-300, 300, out.shape[-1]))
            return TwoHotDist(out, bins, len(self._shape))

        if self._dist == "mse":
            return MSEDist(out, len(self._shape), "sum")

        if self._dist == "normal":
            lo, hi = self._minstd, self._maxstd
            assert std is not None
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            dist = tfd.Normal(jnp.tanh(out), std)
            dist = tfd.Independent(dist, len(self._shape))
            dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist

        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))

        if self._dist == "onehot":
            if self._unimix:
                # if (isinstance(self._unimix, int) and self._unimix) or isinstance(
                #     self._unimix, Tracer
                # ):
                probs = jax.nn.softmax(out, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                unimix_value = self._unimix

                if step is not None:
                    if step.shape[0] != probs.shape[0]:
                        # if step == (bs), val == (bs*bl, d*), then repeat step to match val
                        # step -> (bs, bl) -> (bs, bl) -> (bs*bl)
                        step = step[0, None].repeat(probs.shape[0], axis=0).reshape(-1)

                    step_broadcast = step.reshape(
                        step.shape[0], *([1] * (len(probs.shape) - 1))
                    )
                    if self._unimix_decay != "none":
                        assert isinstance(self._unimix_decay, dict), self._unimix_decay
                        unimix_value = get_unimix_decay(
                            self._unimix_decay["init"],
                            self._unimix_decay["final"],
                            self._unimix_decay["steps"],
                            step_broadcast,
                        )  # same shape as step: (n_devices) or (bs,)

                    unimix_value = jnp.where(
                        step_broadcast == -1, 0, unimix_value
                    )  # (bs, )

                probs = (1 - unimix_value) * probs + unimix_value * uniform
                out = jnp.log(probs)

            dist = OneHotDist(out)
            if len(self._shape) > 1:
                dist = tfd.Independent(dist, len(self._shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
            return dist

        raise NotImplementedError(self._dist)


def kl_div(p, q) -> jax.Array:
    both_gt_zero_mask = lax.bitwise_and(lax.gt(p, 0.0), lax.gt(q, 0.0))
    one_zero_mask = lax.bitwise_and(lax.eq(p, 0.0), lax.ge(q, 0.0))

    one_filler = lax.full_like(p, 1.0)
    inf_filler = lax.full_like(p, np.inf)

    safe_p = lax.select(both_gt_zero_mask, p, one_filler)
    safe_q = lax.select(both_gt_zero_mask, q, one_filler)

    log_val = lax.add(
        lax.sub(lax.mul(safe_p, lax.log(lax.div(safe_p, safe_q))), safe_p), safe_q
    )
    result = lax.select(
        both_gt_zero_mask, log_val, lax.select(one_zero_mask, q, inf_filler)
    )
    return result


CLASS_WEIGHTS = []


def make_image_dist(
    image_dist, mean, task=None, weighted=False, focal=False
):  # mean: bs, bl, 10, 10, 17
    mean = mean.astype(f32)
    if image_dist == "normal":
        return tfd.Independent(tfd.Normal(mean, 1), 3)

    if image_dist == "mse":
        return MSEDist(mean, 3, "sum")

    if image_dist == "mse_max":
        # return jaxutils.MSEMaxDist(mean, 3, "sum")
        raise NotImplementedError

    if image_dist == "abs":
        # return jaxutils.AbsDist(mean, 3, "sum")
        raise NotImplementedError

    if image_dist == "binary":  # for messenger
        if weighted:
            return tfd.Independent(WeightedBernoulli(mean, task), 3)
        elif focal:
            return tfd.Independent(FocalLossBernoulli(mean), 3)
        else:
            # take reduce_sum over the last 3 dims
            return tfd.Independent(tfd.Bernoulli(logits=mean), 3)

    raise NotImplementedError(image_dist)


def get_unimix_decay(init, final, steps, step):
    # decay linearly from init to final over steps, step is the current step
    return jnp.maximum(final, init - (init - final) / steps * step)


def get_linear_increase(init, final, steps, step):
    # increase linearly from init to final over steps, step is the current step
    return jnp.minimum(final, init + (final - init) / steps * step)


def get_linear_increase_kl(init, final, steps, step, step_init=0):
    # increase linearly from init to final over steps, step is the current step
    return jnp.where(
        step < step_init,
        init,
        jnp.minimum(final, init + (final - init) / steps * (step - step_init)),
    )


def get_cubic_ease_in(init, final, steps, step, step_init=0):
    # (step / steps)**3
    return jnp.where(
        step < step_init,
        init,
        jnp.minimum(final, init + (final - init) * (((step - step_init) / steps) ** 3)),
    )
