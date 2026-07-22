#!/usr/bin/env python3
"""
Test CUDA and JAX compatibility to diagnose issues.
"""

import os
import sys
from termcolor import cprint


def test_cuda_setup():
    """Test CUDA and JAX setup."""
    print("=" * 50)
    print("CUDA/JAX Compatibility Test")
    print("=" * 50)

    # Print environment variables
    print("\n1. Environment Variables:")
    for var in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME", "XLA_FLAGS"]:
        value = os.environ.get(var, "NOT SET")
        print(f"   {var}: {value}")

    # Test JAX
    print("\n2. JAX Setup:")
    try:
        import jax
        import jaxlib

        print(f"   JAX version: {jax.__version__}")
        print(f"   JAXlib version: {jaxlib.__version__}")
        print(f"   JAX devices: {jax.devices()}")
        print(f"   JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
        cprint("   ✓ JAX import successful", "green")
    except Exception as e:
        cprint(f"   ✗ JAX import failed: {e}", "red")
        return False

    # Test simple JAX operation
    print("\n3. JAX GPU Operation Test:")
    try:
        import jax.numpy as jnp

        x = jnp.array([1.0, 2.0, 3.0])
        y = x + 1
        print(f"   Simple operation result: {y}")
        cprint("   ✓ Basic JAX operation successful", "green")
    except Exception as e:
        cprint(f"   ✗ Basic JAX operation failed: {e}", "red")
        return False

    # Test broadcasting (the operation that's failing)
    print("\n4. JAX Broadcasting Test:")
    try:
        import jax.numpy as jnp

        shape = (2, 3)
        ones = jnp.ones(shape)
        print(f"   Broadcasting ones with shape {shape}: {ones}")
        cprint("   ✓ Broadcasting operation successful", "green")
    except Exception as e:
        cprint(f"   ✗ Broadcasting operation failed: {e}", "red")
        return False

    # Test GPU memory allocation
    print("\n5. GPU Memory Test:")
    try:
        import jax.numpy as jnp

        large_array = jnp.zeros((1000, 1000))
        result = large_array.sum()
        print(f"   Large array sum: {result}")
        cprint("   ✓ GPU memory allocation successful", "green")
    except Exception as e:
        cprint(f"   ✗ GPU memory allocation failed: {e}", "red")
        return False

    print("\n" + "=" * 50)
    cprint("ALL TESTS PASSED!", "green", attrs=["bold"])
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_cuda_setup()
    sys.exit(0 if success else 1)
