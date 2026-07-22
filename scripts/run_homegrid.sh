#! /bin/bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/activate_ledwm_cuda12.sh"

task=$1
name=$2
device=$3
seed=$4

shift
shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python ledwm/train.py \
  --run.script parallel \
  --logdir ./logdir/homegrid/$name \
  --use_wandb True \
  --task $task \
  --envs.amount 64 \
  --run.actor_batch 64 \
  --seed $seed \
  --encoder.mlp_keys token$ \
  --decoder.mlp_keys token$ \
  --decoder.vector_dist onehot \
  --batch_size 100 \
  --batch_length 256 \
  --run.train_ratio 32 \
  "$@"
