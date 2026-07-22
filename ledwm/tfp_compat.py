"""Compatibility shims for TensorFlow Probability's JAX substrate."""

import warnings


def apply_tfp_jax_compat():
    try:
        from jax import core
        from jax.interpreters import xla
    except Exception:
        return

    if hasattr(xla, "pytype_aval_mappings"):
        return
    if not hasattr(core, "pytype_aval_mappings"):
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        xla.pytype_aval_mappings = core.pytype_aval_mappings


apply_tfp_jax_compat()
