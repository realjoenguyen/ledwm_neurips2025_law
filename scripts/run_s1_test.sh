#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_s1_test.sh --gpus IDS [-- train flags...]

Options:
  --gpus IDS  CUDA device ids, for example 0 or 0,1,2,3.
  -h, --help  Show this help.

Example:
  scripts/run_s1_test.sh --gpus 0 -- --num_eval_eps 50
USAGE
}

die() {
  echo "ERROR: $*" >&2
  echo >&2
  usage >&2
  exit 1
}

task=s1
gpu_ids=
train_args=()

while (($#)); do
  case "$1" in
    --gpus)
      [[ $# -ge 2 ]] || die "--gpus requires a value"
      gpu_ids=$2
      shift 2
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
      die "unexpected positional argument '$1'; use --gpus"
      ;;
  esac
done

[[ -n $gpu_ids ]] || die "--gpus is required"

source "$script_dir/activate_ledwm_cuda12.sh"

export CUDA_VISIBLE_DEVICES=$gpu_ids
python ledwm/train.py \
--run.script parallel_eval \
--logdir ./logdir/messenger/${task}_sent \
--use_wandb False \
--task messenger_${task} \
--env.messenger.length 4 \
--batch_size 100 \
--batch_length 30 \
--envs.amount 30 \
--run.actor_batch 30 \
"${train_args[@]}"
