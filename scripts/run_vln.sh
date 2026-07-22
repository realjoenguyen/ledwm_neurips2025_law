#! /bin/bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/activate_ledwm_cuda12.sh"

name=$1
device=$2
seed=$3

shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python ledwm/train.py \
  --configs vln \
  --seed $seed \
  --run.script parallel \
  --run.train_fill 512 \
  --run.train_ratio 32 \
  --logdir ~/logdir/$name \
  --use_wandb True \
  --envs.amount 4 \
  --run.actor_batch 4 \
  --batch_size 8 \
  --batch_length 256 \
  --env.vln.dataset train \
  --rssm.deter 8192
  # --rssm.bottleneck 1024 \
  "$@"
