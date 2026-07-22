#!/usr/bin/env python3
"""
Quick JAX debug setup - run this before your training for fastest debugging.
Usage: python debug_jax.py && python ledwm/train.py --configs ...
"""

import os
import jax

# Disable JIT compilation completely for debugging
os.environ["JAX_DISABLE_JIT"] = "1"

# Enable all debugging features
os.environ["JAX_DEBUG_NANS"] = "1"
os.environ["JAX_DEBUG_INFS"] = "1"

# Disable XLA optimizations for faster compilation
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes"

# Use less memory for faster allocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

# Enable JAX compilation cache (if supported)
try:
    os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
except:
    pass

print("JAX Debug mode enabled:")
print(f"- JIT disabled: {os.environ.get('JAX_DISABLE_JIT', 'False')}")
print(f"- Debug NANs: {os.environ.get('JAX_DEBUG_NANS', 'False')}")
print(
    f"- Memory fraction: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'default')}"
)

# Test that JAX is working
import jax.numpy as jnp

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Simple test computation
x = jnp.array([1.0, 2.0, 3.0])
print(f"Test computation: {jnp.sum(x)}")
print("\nReady for debugging! 🚀")
