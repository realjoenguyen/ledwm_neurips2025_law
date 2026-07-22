#!/usr/bin/env bash

ledwm_conda_env=ledwm_cuda12

if [[ ${CONDA_DEFAULT_ENV:-} != "$ledwm_conda_env" ]]; then
  conda_exe=${CONDA_EXE:-}
  if [[ -z $conda_exe ]]; then
    conda_exe=$(command -v conda || true)
  fi
  if [[ -z $conda_exe ]]; then
    echo "ERROR: Conda was not found. Install Conda and create the environment with:" >&2
    echo "  conda env create -f environment.yml" >&2
    return 1
  fi

  conda_base=$("$conda_exe" info --base)
  conda_sh=$conda_base/etc/profile.d/conda.sh
  if [[ ! -r $conda_sh ]]; then
    echo "ERROR: Conda initialization script not found: $conda_sh" >&2
    return 1
  fi

  source "$conda_sh"
  conda activate "$ledwm_conda_env"
fi
hash -r

export LEDWM_ACTIVE_CONDA_ENV=$ledwm_conda_env

# Persistent executables are not safe to share across JAX versions, GPU compute
# capabilities, or CUDA toolchains. Keep each environment and GPU architecture
# in a separate namespace. Re-source safely when the current path is one of our
# managed cache directories, while preserving a caller's custom cache path.
case ${JAX_COMPILATION_CACHE_DIR:-} in
  ""|*/ledwm_jax_cache_*_cc*)
    configure_managed_jax_cache=1
    ;;
  *)
    configure_managed_jax_cache=0
    ;;
esac
if ((configure_managed_jax_cache)); then
  gpu_cc=$(
    { nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null || true; } |
      tr -d '. \r' | sort -u | tr '\n' '_' | sed 's/_$//'
  )
  cache_root=${XDG_CACHE_HOME:-$HOME/.cache}
  cache_name=ledwm_jax_cache_${ledwm_conda_env}
  export JAX_COMPILATION_CACHE_DIR="$cache_root/${cache_name}_cc${gpu_cc:-unknown}"
fi
