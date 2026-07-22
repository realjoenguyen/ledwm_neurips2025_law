#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
source "$script_dir/activate_ledwm_cuda12.sh"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_s2.sh --gpus IDS --batch-size N|auto [options] [-- train flags...]

Options:
  --gpus IDS          CUDA device ids, for example 0 or 0,1,2,3.
  --batch-size N      Batch size, or auto to probe and cache the GPU-aware limit.
  --auto-batch-min N  Smallest global batch to probe (default: 128 per GPU).
  --auto-batch-max N  Largest global batch to probe (default: 4096 per GPU).
  --auto-batch-quantum N
                      Probe granularity (default: 128 per GPU).
  --auto-batch-safety PCT
                      Percent of the largest fitting batch to use (default: 90).
  --server LABEL      Optional metadata passed as --run.server LABEL.
  --configs NAMES     Config stack. If omitted, uses the default S2 train stack.
  --preset NAME       Append one named config to the config stack.
  --resume            Resume the newest run with model, optimizer, and replay state.
  --log-level LEVEL   Loguru level: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR,
                      or CRITICAL (default: LOGURU_LEVEL or INFO).
  --dry-run           Print the resolved command without running training.
  -h, --help          Show this help.

Examples:
  scripts/run_s2.sh --gpus 0 --batch-size 40
  scripts/run_s2.sh --gpus 1 --batch-size auto -- --batch_length 60
  scripts/run_s2.sh --gpus 0,1 --batch-size 80 --server dgx -- --run.from_checkpoint ckpt
USAGE
}

die() {
  echo "ERROR: $*" >&2
  echo >&2
  usage >&2
  exit 1
}

append_words() {
  local value=$1
  local word
  read -r -a words <<<"$value"
  for word in "${words[@]}"; do
    configs+=("$word")
  done
}

default_configs=(
  s2_train
)

gpu_ids=
batch_size=
server=
resume=0
dry_run=${DRY_RUN:-0}
log_level=${LOGURU_LEVEL:-INFO}
log_enqueue=${LOGURU_ENQUEUE:-1}
configs=()
presets=()
train_args=()
jax_mem_fraction=
jax_prealloc=
jax_allocator=
jax_xla_autotune_level=
jax_quiet_xla=
auto_batch_min=
auto_batch_max=
auto_batch_quantum=
auto_batch_safety=90

while (($#)); do
  case "$1" in
    --gpus)
      [[ $# -ge 2 ]] || die "--gpus requires a value"
      gpu_ids=$2
      shift 2
      ;;
    --batch-size|--batch_size)
      [[ $# -ge 2 ]] || die "--batch-size requires a value"
      batch_size=$2
      shift 2
      ;;
    --auto-batch-min)
      [[ $# -ge 2 ]] || die "--auto-batch-min requires a value"
      auto_batch_min=$2
      shift 2
      ;;
    --auto-batch-max)
      [[ $# -ge 2 ]] || die "--auto-batch-max requires a value"
      auto_batch_max=$2
      shift 2
      ;;
    --auto-batch-quantum)
      [[ $# -ge 2 ]] || die "--auto-batch-quantum requires a value"
      auto_batch_quantum=$2
      shift 2
      ;;
    --auto-batch-safety)
      [[ $# -ge 2 ]] || die "--auto-batch-safety requires a value"
      auto_batch_safety=$2
      shift 2
      ;;
    --server)
      [[ $# -ge 2 ]] || die "--server requires a value"
      server=$2
      shift 2
      ;;
    --resume)
      resume=1
      shift
      ;;
    --log-level)
      [[ $# -ge 2 ]] || die "--log-level requires a value"
      log_level=$2
      shift 2
      ;;
    --configs)
      shift
      [[ $# -gt 0 && $1 != --* ]] || die "--configs requires at least one config name"
      while (($#)) && [[ $1 != --* ]]; do
        append_words "$1"
        shift
      done
      ;;
    --preset)
      [[ $# -ge 2 ]] || die "--preset requires a value"
      presets+=("$2")
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --)
      shift
      train_args+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      train_args+=("$@")
      break
      ;;
    *)
      die "unexpected positional argument '$1'; use --gpus and --batch-size"
      ;;
  esac
done

[[ -n $gpu_ids ]] || die "--gpus is required"
[[ -n $batch_size ]] || die "--batch-size is required"
[[ $batch_size == auto || $batch_size =~ ^[0-9]+$ ]] ||
  die "--batch-size must be an integer or 'auto'"
log_level=${log_level^^}
case $log_level in
  TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL) ;;
  *) die "--log-level must be TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, or CRITICAL" ;;
esac

if ((${#configs[@]} == 0)); then
  configs=("${default_configs[@]}")
fi
configs+=("${presets[@]}")

num_gpus=$(tr -cd ',' <<<"$gpu_ids" | wc -c)
num_gpus=$((num_gpus + 1))

resolve_auto_batch_size() {
  local quantum=${auto_batch_quantum:-$((128 * num_gpus))}
  local minimum=${auto_batch_min:-$quantum}
  local maximum=${auto_batch_max:-$((4096 * num_gpus))}
  local candidate candidate_units high_units last_fit_units mid_units
  local max_fit selected probe_root probe_log
  local gpu_set_signature code_revision dirty_hash request_hash cache_root cache_file
  local used
  local arg skip_shape_value=0
  local shape_args=()

  ((maximum > 20000)) && maximum=20000
  [[ $quantum =~ ^[0-9]+$ && $quantum -gt 0 ]] ||
    die "--auto-batch-quantum must be a positive integer"
  [[ $minimum =~ ^[0-9]+$ && $minimum -gt 0 ]] ||
    die "--auto-batch-min must be a positive integer"
  [[ $maximum =~ ^[0-9]+$ && $maximum -gt 0 ]] ||
    die "--auto-batch-max must be a positive integer"
  [[ $auto_batch_safety =~ ^[0-9]+$ ]] ||
    die "--auto-batch-safety must be an integer percentage"
  ((auto_batch_safety >= 1 && auto_batch_safety <= 100)) ||
    die "--auto-batch-safety must be between 1 and 100"
  ((quantum % num_gpus == 0)) ||
    die "--auto-batch-quantum must be divisible by the number of GPUs"

  minimum=$((((minimum + quantum - 1) / quantum) * quantum))
  maximum=$(((maximum / quantum) * quantum))
  ((minimum <= maximum)) || die "auto batch minimum exceeds maximum"

  while IFS= read -r used; do
    used=${used// /}
    ((used <= 1024)) || die \
      "auto batch probing requires idle GPUs; selected GPU is using ${used} MiB"
  done < <(
    nvidia-smi -i "$gpu_ids" --query-gpu=memory.used --format=csv,noheader,nounits
  )

  # Key the result to the physical GPU set, rather than only its model and
  # count. Sorting makes the same set reusable regardless of --gpus ordering.
  gpu_set_signature=$(nvidia-smi -i "$gpu_ids" \
    --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits |
    LC_ALL=C sort | tr '\n' ';')
  code_revision=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
  dirty_hash=$(git -C "$repo_root" diff --no-ext-diff --binary -- ledwm scripts/run_s2.sh |
    sha256sum | awk '{print $1}')
  for arg in "${train_args[@]}"; do
    if ((skip_shape_value)); then
      skip_shape_value=0
      continue
    fi
    case "$arg" in
      --run.compile_only|--jax.profiler|--logdir|--use_wandb)
        skip_shape_value=1
        ;;
      --run.compile_only=*|--jax.profiler=*|--logdir=*|--use_wandb=*)
        ;;
      *)
        shape_args+=("$arg")
        ;;
    esac
  done
  request_hash=$(printf '%s\n' \
    s2 "$gpu_set_signature" "$num_gpus" "$code_revision" "$dirty_hash" \
    "${configs[*]}" "${shape_args[*]}" \
    "${LEDWM_FAST_TRAIN_METRICS:-1}" "${LEDWM_FAST_OPTIMIZER_METRICS:-1}" \
    "${LEDWM_SKIP_ADAM_METRICS:-1}" \
    "${LEDWM_SKIP_TRAIN_OUTS:-1}" | sha256sum | awk '{print $1}')
  cache_root=${LEDWM_AUTOBATCH_CACHE_DIR:-${JAX_COMPILATION_CACHE_DIR%/*}/ledwm_s2_autobatch}
  cache_file="$cache_root/$request_hash.max_batch"

  if [[ -s $cache_file ]] && read -r max_fit <"$cache_file" &&
    [[ $max_fit =~ ^[0-9]+$ ]] && ((max_fit >= minimum && max_fit <= maximum)); then
    echo "autobatch.cache_hit | max_fit=$max_fit | path=$cache_file" >&2
  else
    [[ $dry_run != 1 && $dry_run != true ]] ||
      die "auto batch cache miss during --dry-run; run once without --dry-run"
    mkdir -p "$cache_root"
    probe_root=$(mktemp -d "${TMPDIR:-/tmp}/ledwm-autobatch.XXXXXXXX")

    probe_batch() {
      local value=$1
      local probe=(bash "$script_dir/run_s2.sh" --gpus "$gpu_ids" --batch-size "$value")
      probe+=(--configs "${configs[@]}")
      [[ -z $server ]] || probe+=(--server "$server")
      probe+=(-- "${train_args[@]}" --run.compile_only true --logdir "$probe_root")
      probe_log="$probe_root/probe-$value.log"
      echo "autobatch.probe | batch_size=$value | gpus=$gpu_ids" >&2
      DRY_RUN=0 WANDB_MODE=disabled "${probe[@]}" >"$probe_log" 2>&1
    }

    # Probe the upper bound first. A fitting maximum resolves the search in one
    # compile; otherwise binary search the remaining monotonic fit range.
    high_units=$((maximum / quantum))
    last_fit_units=0
    candidate=$maximum
    if probe_batch "$candidate"; then
      last_fit_units=$high_units
      echo "autobatch.fit | batch_size=$candidate" >&2
    else
      echo "autobatch.oom_or_error | batch_size=$candidate | log=$probe_log" >&2
      candidate_units=$((minimum / quantum))
      high_units=$((high_units - 1))
      while ((candidate_units <= high_units)); do
        mid_units=$(((candidate_units + high_units) / 2))
        candidate=$((mid_units * quantum))
        if probe_batch "$candidate"; then
          last_fit_units=$mid_units
          candidate_units=$((mid_units + 1))
          echo "autobatch.fit | batch_size=$candidate" >&2
        else
          high_units=$((mid_units - 1))
          echo "autobatch.oom_or_error | batch_size=$candidate | log=$probe_log" >&2
        fi
      done
    fi
    if ((last_fit_units == 0)); then
      tail -n 20 "$probe_log" >&2
      rm -rf -- "$probe_root"
      die "smallest auto batch $minimum did not fit"
    fi
    max_fit=$((last_fit_units * quantum))
    printf '%s\n' "$max_fit" >"$cache_file"
    rm -rf -- "$probe_root"
    echo "autobatch.cache_written | max_fit=$max_fit | path=$cache_file" >&2
  fi

  selected=$((((max_fit * auto_batch_safety / 100) / quantum) * quantum))
  ((selected >= minimum)) || selected=$minimum
  echo "autobatch.selected | batch_size=$selected | max_fit=$max_fit | safety=${auto_batch_safety}%" >&2
  batch_size=$selected
}

if [[ $batch_size == auto ]]; then
  resolve_auto_batch_size
fi

if ((batch_size % num_gpus != 0)); then
  echo "ERROR: batch-size ($batch_size) is not divisible by the number of GPUs ($num_gpus)." >&2
  exit 1
fi

seq_ids=$(seq -s ',' 0 $((num_gpus - 1)))

cmd=(
  python ledwm/train.py
  --configs "${configs[@]}"
  --jax.train_devices "$seq_ids"
  --jax.policy_devices 0
  --batch_size "$batch_size"
)

if [[ -n $server ]]; then
  cmd+=(--run.server "$server")
fi
if ((resume)); then
  cmd+=(--resume)
fi

cmd+=("${train_args[@]}")

for ((i = 0; i < ${#train_args[@]}; i++)); do
  arg=${train_args[$i]}
  case "$arg" in
    --jax.mem_fraction=*)
      jax_mem_fraction=${arg#*=}
      ;;
    --jax.mem_fraction)
      if ((i + 1 < ${#train_args[@]})); then
        jax_mem_fraction=${train_args[$((i + 1))]}
      fi
      ;;
    --jax.prealloc=*)
      jax_prealloc=${arg#*=}
      ;;
    --jax.prealloc)
      if ((i + 1 < ${#train_args[@]})); then
        jax_prealloc=${train_args[$((i + 1))]}
      fi
      ;;
    --jax.allocator=*)
      jax_allocator=${arg#*=}
      ;;
    --jax.allocator)
      if ((i + 1 < ${#train_args[@]})); then
        jax_allocator=${train_args[$((i + 1))]}
      fi
      ;;
    --jax.xla_autotune_level=*)
      jax_xla_autotune_level=${arg#*=}
      ;;
    --jax.xla_autotune_level)
      if ((i + 1 < ${#train_args[@]})); then
        jax_xla_autotune_level=${train_args[$((i + 1))]}
      fi
      ;;
    --jax.quiet_xla=*)
      jax_quiet_xla=${arg#*=}
      ;;
    --jax.quiet_xla)
      if ((i + 1 < ${#train_args[@]})); then
        jax_quiet_xla=${train_args[$((i + 1))]}
      fi
      ;;
  esac
done

if [[ ${jax_allocator,,} == platform ]]; then
  jax_xla_autotune_level=${jax_xla_autotune_level:-0}
  jax_quiet_xla=${jax_quiet_xla:-true}
fi

jax_env=()
xla_flags=${XLA_FLAGS:-}
jax_env+=("LOGURU_LEVEL=$log_level")
jax_env+=("LOGURU_ENQUEUE=$log_enqueue")
if [[ -n $jax_mem_fraction ]]; then
  jax_env+=("XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_mem_fraction")
fi
if [[ -n $jax_allocator ]]; then
  jax_env+=("XLA_PYTHON_CLIENT_ALLOCATOR=$jax_allocator")
fi
if [[ -n $jax_xla_autotune_level && $jax_xla_autotune_level != -1 ]]; then
  xla_flags="${xla_flags:+$xla_flags }--xla_gpu_autotune_level=$jax_xla_autotune_level"
  jax_env+=("XLA_FLAGS=$xla_flags")
fi
case "${jax_quiet_xla,,}" in
  true|1|yes)
    jax_env+=("TF_CPP_MIN_LOG_LEVEL=2")
    ;;
esac
case "${jax_prealloc,,}" in
  false|0|no)
    jax_env+=("XLA_PYTHON_CLIENT_PREALLOCATE=false")
    ;;
esac

if [[ $dry_run == 1 || $dry_run == true ]]; then
  for env_var in "${jax_env[@]}"; do
    echo "$env_var"
  done
  echo "CUDA_VISIBLE_DEVICES=$gpu_ids"
  printf 'COMMAND='
  printf ' %s' "${cmd[@]}"
  printf '\n'
  printf 'RUN='
  for env_var in "${jax_env[@]}"; do
    printf '%s ' "$env_var"
  done
  printf 'CUDA_VISIBLE_DEVICES=%s' "$gpu_ids"
  printf ' %s' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ -n $jax_mem_fraction ]]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_mem_fraction
fi
if [[ -n $jax_allocator ]]; then
  export XLA_PYTHON_CLIENT_ALLOCATOR=$jax_allocator
fi
if [[ -n $jax_xla_autotune_level && $jax_xla_autotune_level != -1 ]]; then
  export XLA_FLAGS=$xla_flags
fi
case "${jax_quiet_xla,,}" in
  true|1|yes)
    export TF_CPP_MIN_LOG_LEVEL=2
    ;;
esac
case "${jax_prealloc,,}" in
  false|0|no)
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    ;;
esac
export CUDA_VISIBLE_DEVICES=$gpu_ids
export LOGURU_LEVEL=$log_level
export LOGURU_ENQUEUE=$log_enqueue
exec "${cmd[@]}"
