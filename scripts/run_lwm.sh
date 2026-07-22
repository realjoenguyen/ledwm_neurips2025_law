#!/usr/bin/env bash
set -euo pipefail

scripts_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$scripts_dir/.." && pwd)

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_lwm.sh --task LEVEL --gpus IDS --batch-size N|auto [options] [-- train flags...]
  scripts/run_lwm.sh GPU LEVEL [train flags...]  # legacy Messenger-LWM interface

Arguments:
  --task LEVEL         Messenger-LWM split: easy, medium, or hard.
  --gpus IDS           CUDA device ids, for example 0 or 0,1,2,3.
  --batch-size N       Global batch size, or auto to probe the GPU-aware limit.

All run_s2.sh launcher options are supported, including --auto-batch-min,
--auto-batch-max, --auto-batch-quantum, --auto-batch-safety, --configs,
--preset, --resume, --server, --log-level, and --dry-run.

Examples:
  scripts/run_lwm.sh --task easy --gpus 0 --batch-size 40
  scripts/run_lwm.sh --task hard --gpus 1 --batch-size auto -- --batch_length 60
USAGE
}

die() {
  echo "ERROR: $*" >&2
  echo >&2
  usage >&2
  exit 1
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  usage
  exit 0
fi

if (($#)) && [[ $1 != --* ]]; then
  # Preserve the original Messenger-LWM GPU/LEVEL interface.
  if [[ ${2:-} == easy || ${2:-} == medium || ${2:-} == hard ]]; then
    device=$1
    task=$2
    shift 2
    source "$scripts_dir/activate_ledwm_cuda12.sh"
    export CUDA_VISIBLE_DEVICES=$device
    exec python ledwm/train.py \
      --task "lwm_${task}" \
      --rssm.task "$task" \
      "$@"
  fi

  # Preserve run_lwm.sh's original six-position interface for run.sh.
  (($# >= 6)) || die \
    "legacy usage requires SERVER BATCH_SIZE GPUS TASK CONFIGS TRAIN_FLAGS"
  server=$1
  batch_size=$2
  gpu_ids=$3
  task=$4
  read -r -a configs <<<"$5"
  read -r -a train_args <<<"$6"
  shift 6

  exec bash "$scripts_dir/run_lwm.sh" \
    --task "$task" \
    --gpus "$gpu_ids" \
    --batch-size "$batch_size" \
    --server "$server" \
    --configs "${configs[@]}" lwm \
    -- \
    "${train_args[@]}" \
    "$@"
fi

task=
has_gpus=0
has_batch_size=0
has_configs=0
launcher_args=()
train_args=()

while (($#)); do
  case "$1" in
    --task)
      [[ $# -ge 2 ]] || die "--task requires easy, medium, or hard"
      task=$2
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || die "--gpus requires a value"
      launcher_args+=("$1" "$2")
      has_gpus=1
      shift 2
      ;;
    --batch-size|--batch_size)
      [[ $# -ge 2 ]] || die "--batch-size requires a value"
      launcher_args+=("$1" "$2")
      has_batch_size=1
      shift 2
      ;;
    --auto-batch-min|--auto-batch-max|--auto-batch-quantum|--auto-batch-safety|--server|--log-level|--preset)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      launcher_args+=("$1" "$2")
      shift 2
      ;;
    --configs)
      launcher_args+=("$1")
      has_configs=1
      shift
      [[ $# -gt 0 && $1 != --* ]] || die "--configs requires at least one config name"
      while (($#)) && [[ $1 != --* ]]; do
        launcher_args+=("$1")
        shift
      done
      ;;
    --resume|--dry-run)
      launcher_args+=("$1")
      shift
      ;;
    --)
      shift
      train_args+=("$@")
      break
      ;;
    --*)
      # Forward training flags without hiding launcher options that appear later.
      # Use an explicit `--` above when every remaining argument must be passed
      # through verbatim, including names reserved by this launcher.
      train_args+=("$1")
      shift
      ;;
    *)
      if ((${#train_args[@]})); then
        train_args+=("$1")
        shift
      else
        die "unexpected positional argument '$1'"
      fi
      ;;
  esac
done

case $task in
  easy|medium|hard) ;;
  "") die "--task is required" ;;
  *) die "--task must be easy, medium, or hard" ;;
esac
((has_gpus)) || die "--gpus is required"
((has_batch_size)) || die "--batch-size is required"

# Reuse the shared capacity-probe implementation with the complete LWM training
# stack defined alongside the other task configs.
if ((!has_configs)); then
  launcher_args+=(--configs lwm_train)
fi

exec bash "$scripts_dir/run_s2.sh" \
  "${launcher_args[@]}" \
  -- \
  --task "lwm_${task}" \
  --rssm.task "$task" \
  "${train_args[@]}"
