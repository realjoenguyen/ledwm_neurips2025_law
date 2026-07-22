#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_lwm_test.sh --task LEVEL --gpus IDS [options] [-- train flags...]

Options:
  --task LEVEL     Messenger-LWM split: easy, medium, or hard.
  --gpus IDS       CUDA device ids, for example 0 or 0,1,2,3.
  --batch-size N   Global batch size (default: 100).
  --dry-run        Print the resolved command without running it.
  -h, --help       Show this help.

Example:
  scripts/run_lwm_test.sh --task hard --gpus 0 -- --run.from_checkpoint ./checkpoint.ckpt
USAGE
}

die() {
  echo "ERROR: $*" >&2
  echo >&2
  usage >&2
  exit 1
}

task=
gpu_ids=
batch_size=100
dry_run=0
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
      gpu_ids=$2
      shift 2
      ;;
    --batch-size|--batch_size)
      [[ $# -ge 2 ]] || die "--batch-size requires a value"
      batch_size=$2
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
      die "unexpected positional argument '$1'; use --task and --gpus"
      ;;
  esac
done

case $task in
  easy|medium|hard) ;;
  "") die "--task is required" ;;
  *) die "--task must be easy, medium, or hard" ;;
esac
[[ -n $gpu_ids ]] || die "--gpus is required"
[[ $batch_size =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"

command=(
  bash "$script_dir/run_lwm.sh"
  --task "$task"
  --gpus "$gpu_ids"
  --batch-size "$batch_size"
)
((dry_run)) && command+=(--dry-run)
command+=(
  --
  --run.script parallel_eval
  --logdir "./logdir/messenger/lwm_${task}_sent"
  --use_wandb False
  --env.lwm.length 64
  --batch_length 256
  --envs.amount 1
  --run.actor_batch 1
  "${train_args[@]}"
)

exec "${command[@]}"
