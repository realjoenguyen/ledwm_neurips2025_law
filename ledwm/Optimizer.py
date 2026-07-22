import os

from termcolor import cprint
import wandb
from ledwm import ninjax as nj

from . import jaxutils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import re


def skip_adam_metrics_enabled():
    return os.environ.get("LEDWM_SKIP_ADAM_METRICS") == "1"


def fast_optimizer_metrics_enabled():
    return os.environ.get("LEDWM_FAST_OPTIMIZER_METRICS") == "1"


class Optimizer(nj.Module):
    PARAM_COUNTS = {}

    def __init__(
        self,
        lr,
        opt="adam",
        eps=1e-8,
        clip=100.0,
        warmup=0,
        wd=0.0,
        # wd_pattern=r"/(w|kernel)$",
        lateclip=0.0,
        frozen_keys=r"^$",
        clip_grad_atten=0.0,
        clip_grad_encoder=0.0,
        lr_atten=0.0,
        lr_encoder=0.0,
        not_optimize=False,  # to record grad only
        beta1=0.9,
        beta2=0.999,
        config=None,
    ):
        # assert wd_pattern[0] not in ("0", "1")
        # assert self.path not in self.PARAM_COUNTS
        if not_optimize:  # record grad only
            lr = 0.0

        self.PARAM_COUNTS[self.path] = None
        # wd_pattern = re.compile(wd_pattern)
        frozen_keys = re.compile(frozen_keys)
        self.config = config

        # Store eps for later use in metrics computation
        self._adam_eps = eps

        def get_optimizer(clip, lr):
            chain = []
            if clip > 0:
                chain.append(optax.clip_by_global_norm(clip))
            if opt == "adam":
                chain.append(optax.scale_by_adam(eps=eps, b1=beta1, b2=beta2))

            elif opt == "lion":
                chain.append(optax.scale_by_lion())
            else:
                raise NotImplementedError(opt)

            if lateclip:
                chain.append(jaxutils.late_grad_clip(lateclip))

            if wd > 0:
                chain.append(optax.add_decayed_weights(wd))

            if warmup:
                schedule = optax.linear_schedule(-0.001, -lr, warmup)
                chain.append(optax.inject_hyperparams(optax.scale)(schedule))
            else:
                chain.append(optax.scale(-lr))

            return optax.chain(*chain)

        def partition_params_fn(params):
            from ledwm.nets.EncoderRSSM import ATTEN_WEIGHT_KEY
            from ledwm.RSSM import ENCODER_RSSM_KEY

            def label_key(k):
                if clip_grad_atten > 0 and ATTEN_WEIGHT_KEY in k:
                    label = "smaller_lr_atten"
                elif clip_grad_encoder > 0 and ENCODER_RSSM_KEY in k:
                    label = "smaller_lr_encoder"
                else:
                    label = "frozen" if bool(frozen_keys.search(k)) else "trainable"

                if label == "frozen":
                    cprint(f"optimizer.parameter | name={k} | group=frozen", "red")
                elif label == "smaller_lr_atten":
                    cprint(
                        f"optimizer.parameter | name={k} | group=attention", "yellow"
                    )
                elif label == "smaller_lr_encoder":
                    cprint(
                        f"optimizer.parameter | name={k} | group=encoder", "yellow"
                    )
                else:
                    cprint(f"optimizer.parameter | name={k} | group=trainable")
                return label

            return jaxutils.tree_map(label_key, jaxutils.tree_keys(params))

        opt_dict = {"trainable": get_optimizer(clip, lr), "frozen": optax.set_to_zero()}
        if clip_grad_atten > 0 or lr_atten > 0:
            cprint(
                f"optimizer.group | name=attention | clip_grad={clip_grad_atten} | "
                f"learning_rate={lr_atten}",
                "red",
            )
            opt_dict["smaller_lr_atten"] = get_optimizer(
                clip_grad_atten if clip_grad_atten > 0 else clip,
                lr_atten if lr_atten > 0 else lr,
            )
        if clip_grad_encoder > 0 or lr_encoder > 0:
            cprint(
                f"optimizer.group | name=encoder | clip_grad={clip_grad_encoder} | "
                f"learning_rate={lr_encoder}",
                "red",
            )
            opt_dict["smaller_lr_encoder"] = get_optimizer(
                clip_grad_encoder if clip_grad_encoder > 0 else clip,
                lr_encoder if lr_encoder > 0 else lr,
            )

        self.opt = optax.multi_transform(opt_dict, partition_params_fn)  # type: ignore
        self.step = nj.Variable(jnp.array, 0, jnp.int32, name="step")
        self.scaling = jaxutils.COMPUTE_DTYPE == jnp.float16
        cprint(
            f"optimizer.config | name={self.name} | "
            f"loss_scaling={str(self.scaling).lower()}",
            "yellow",
        )

        if self.scaling:
            self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
            self.grad_scale = nj.Variable(
                jnp.array, 1e4, jnp.float32, name="grad_scale"
            )
            self.good_steps = nj.Variable(jnp.array, 0, jnp.int32, name="good_steps")

    def reset(self):
        self.step.write(0)
        if self.scaling:
            self.grad_scale.write(1e4)
            self.good_steps.write(0)

    def __call__(self, modules, loss_fn, *args, has_aux=False, **kwargs):
        # modules: parameters - weights
        # if use_opt_step:
        #     cprint("Using opt step", "red")

        def wrapped(*args, **kwargs):
            outs = loss_fn(*args, **kwargs)
            loss, aux = outs if has_aux else (outs, None)
            assert loss.dtype == jnp.float32, (self.name, loss.dtype)
            assert loss.shape == (), (self.name, loss.shape)
            if self.scaling:
                loss *= jaxutils.sg(self.grad_scale.read())
            return loss, aux

        metrics = {}
        loss, params, grads, aux = nj.grad(wrapped, modules, has_aux=True)(  # type: ignore
            *args, **kwargs
        )

        if not self.PARAM_COUNTS[self.path]:
            count = sum([np.prod(x.shape) for x in jaxutils.tree_leaves(params)])
            cprint(
                f"optimizer.variables | name={self.name} | count={count}", "green"
            )
            if wandb.run is not None:
                metric = {f"{self.name}_param_count": count}
                # wandb.config.update(metric, allow_val_change=True)
                wandb.run.summary.update(metric)
            self.PARAM_COUNTS[self.path] = count

        if jaxutils.parallel():
            grads = jaxutils.tree_map(lambda x: jax.lax.pmean(x, "i"), grads)

        if self.scaling:
            grads = jaxutils.tree_map(lambda x: x / self.grad_scale.read(), grads)
            finite = self._update_scale(grads)
            metrics[f"{self.name}_grad_scale"] = self.grad_scale.read()
            metrics[f"{self.name}_grad_overflow"] = (~finite).astype(jnp.float32)

        optstate = self.get("state", self.opt.init, params)
        updates, optstate = self.opt.update(grads, optstate, params)

        self.put("state", optstate)
        nj.context().update(optax.apply_updates(params, updates))

        if fast_optimizer_metrics_enabled():
            # Global gradient norms below are diagnostics only. Gradient
            # clipping already happens inside self.opt.update(), so computing
            # them again makes three full extra parameter-tree reductions per
            # learner update without changing the optimizer result.
            finite_step = finite if self.scaling else jnp.array(True)
            self.step.write(
                self.step.read() + finite_step.astype(jnp.int32)
            )
            metrics["loss"] = loss.mean()
            metrics["grad_steps"] = self.step.read()
            metrics = {f"{self.name}_{key}": value for key, value in metrics.items()}
            return (metrics, aux) if has_aux else metrics

        # Record m / sqrt(v) from Adam optimizer
        adam_metrics = {}
        if not skip_adam_metrics_enabled():
            adam_metrics = self._record_adam_metrics(optstate, grads)

        norm = optax.global_norm(grads)
        # keep all keys that contain 'enc_atten'
        # because enc_atten doesn't have any nested structure so we can directly this
        atten_grads = {k: v for k, v in grads.items() if "enc_atten" in k}
        # assert len(atten_grads) > 0, "No atten grads"
        atten_norm = optax.global_norm(atten_grads)
        atten_pos_norm = 0

        if (
            self.config is not None
            and self.config.rssm.atten.pos_head_seperate
            and len(atten_grads) > 0
        ):
            ATTEN_POS_KEYS = ["query_pos", "key_pos", "value_pos"]
            atten_pos_grads = {
                k: v
                for k, v in grads.items()
                if any(key in k for key in ATTEN_POS_KEYS)
            }
            assert len(atten_pos_grads) > 0, "No atten pos grads"
            for k, v in atten_pos_grads.items():
                cprint(f"optimizer.gradient | name={k} | shape={v.shape}", "yellow")
            atten_pos_norm = optax.global_norm(atten_pos_grads)

        if self.scaling:
            norm = jnp.where(jnp.isfinite(norm), norm, jnp.nan)
            atten_norm = jnp.where(jnp.isfinite(atten_norm), atten_norm, jnp.nan)
            atten_pos_norm = jnp.where(
                jnp.isfinite(atten_pos_norm), atten_pos_norm, jnp.nan
            )
        # record
        # record
        # key_values_grad = collect_immediate_key_values(grads)
        # for k, v in key_values_grad.items():
        #     metrics[f"{self.name}_grad_{k}"] = v.reshape(-1)

        self.step.write(self.step.read() + jnp.isfinite(norm).astype(jnp.int32))
        metrics["loss"] = loss.mean()
        metrics["grad_norm"] = norm  # before clip
        metrics["atten_norm"] = atten_norm
        metrics["atten_pos_norm"] = atten_pos_norm
        metrics["isfinite_norm"] = jnp.isfinite(norm)
        metrics["grad_steps"] = self.step.read()
        metrics = {f"{self.name}_{k}": v for k, v in metrics.items()}

        # Add Adam-specific metrics
        metrics.update({f"{self.name}_{k}": v for k, v in adam_metrics.items()})

        return (metrics, aux) if has_aux else metrics

    def _update_scale(self, grads):
        finite = jnp.array(
            [jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(grads)]
        ).all()
        keep = finite & (self.good_steps.read() < 1000)
        incr = finite & (self.good_steps.read() >= 1000)
        decr = ~finite
        self.good_steps.write(keep.astype(jnp.int32) * (self.good_steps.read() + 1))
        self.grad_scale.write(
            jnp.clip(
                keep.astype(jnp.float32) * self.grad_scale.read()
                + incr.astype(jnp.float32) * self.grad_scale.read() * 2
                + decr.astype(jnp.float32) * self.grad_scale.read() / 2,
                1e-4,
                1e4,
            )
        )
        return finite

    def _record_adam_metrics(self, optstate, grads):
        """Record m / sqrt(v) metrics from Adam optimizer state."""
        adam_metrics = {}

        def extract_adam_state(state, prefix=""):
            """Recursively extract Adam state from multi-transform optimizer."""
            if hasattr(state, "inner_state") and isinstance(state.inner_state, dict):
                # Handle multi_transform case
                for key, inner_state in state.inner_state.items():
                    if key in ["trainable", "smaller_lr_trainable"]:
                        adam_metrics.update(
                            extract_adam_state(inner_state, f"{prefix}{key}_")
                        )
            elif hasattr(state, "__len__") and len(state) > 0:
                # Handle chain of transformations
                for i, transform_state in enumerate(state):
                    if hasattr(transform_state, "mu") and hasattr(
                        transform_state, "nu"
                    ):
                        # Found Adam state (ScaleByAdamState)
                        mu = transform_state.mu  # First moment
                        nu = transform_state.nu  # Second moment

                        # Compute m / sqrt(v + eps) for each parameter
                        # Use the same eps as configured in the optimizer
                        eps = getattr(self, "_adam_eps", 1e-8)  # fallback to default
                        m_over_sqrt_v = jaxutils.tree_map(
                            lambda m, v: m / (jnp.sqrt(v) + eps), mu, nu
                        )

                        # Compute global norm of m / sqrt(v)
                        m_over_sqrt_v_norm = optax.global_norm(m_over_sqrt_v)
                        adam_metrics[f"{prefix}adam_m_over_sqrt_v_norm"] = (
                            m_over_sqrt_v_norm
                        )

                        # Also record first and second moment norms separately
                        adam_metrics[f"{prefix}adam_first_moment_norm"] = (
                            optax.global_norm(mu)
                        )
                        adam_metrics[f"{prefix}adam_second_moment_norm"] = (
                            optax.global_norm(nu)
                        )

                        # Optional: Record per-parameter m/sqrt(v) statistics if needed
                        # Uncomment below for more detailed metrics:
                        # m_over_sqrt_v_values = jaxutils.tree_leaves(m_over_sqrt_v)
                        # if m_over_sqrt_v_values:
                        #     flattened = jnp.concatenate([jnp.reshape(x, -1) for x in m_over_sqrt_v_values])
                        #     adam_metrics[f"{prefix}adam_m_over_sqrt_v_mean"] = jnp.mean(flattened)
                        #     adam_metrics[f"{prefix}adam_m_over_sqrt_v_std"] = jnp.std(flattened)
                        #     adam_metrics[f"{prefix}adam_m_over_sqrt_v_max"] = jnp.max(jnp.abs(flattened))

            return adam_metrics

        return extract_adam_state(optstate)


# def collect_immediate_key_values(pytree):
#     key_values = {}

#     def is_nested(x):
#         return isinstance(x, (dict, list, tuple))

#     def recurse(subtree):
#         if isinstance(subtree, dict):
#             for k, v in subtree.items():
#                 if not is_nested(v):  # Check if v is a leaf node
#                     key_values[k] = v
#                 else:
#                     recurse(v)

#     recurse(pytree)
#     return key_values
