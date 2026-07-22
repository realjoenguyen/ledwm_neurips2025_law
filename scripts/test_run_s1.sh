#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local haystack=$1
  local needle=$2
  [[ "$haystack" == *"$needle"* ]] || fail "expected output to contain: $needle"
}

assert_not_contains() {
  local haystack=$1
  local needle=$2
  [[ "$haystack" != *"$needle"* ]] || fail "expected output not to contain: $needle"
}

output=$(DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0 \
  --batch-size 40)

assert_contains "$output" "--configs s1_train"
assert_contains "$output" "CONDA_ENV=ledwm_cuda12"
assert_contains "$output" "TF_CPP_MIN_LOG_LEVEL=2"
assert_contains "$output" "TF_ENABLE_ONEDNN_OPTS=0"
assert_contains "$output" "LEDWM_FAST_TRAIN_METRICS=1"
assert_contains "$output" "LEDWM_FAST_OPTIMIZER_METRICS=1"
assert_contains "$output" "LEDWM_SKIP_ADAM_METRICS=1"
assert_contains "$output" "LEDWM_SKIP_TRAIN_OUTS=1"
assert_contains "$output" "LOGURU_LEVEL=INFO"
assert_contains "$output" "LOGURU_ENQUEUE=1"
assert_contains "$output" "RUN=TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 PYTHONHASHSEED=0 LEDWM_FAST_TRAIN_METRICS=1 LEDWM_FAST_OPTIMIZER_METRICS=1 LEDWM_SKIP_ADAM_METRICS=1 LEDWM_SKIP_TRAIN_OUTS=1 LOGURU_LEVEL=INFO LOGURU_ENQUEUE=1 CUDA_VISIBLE_DEVICES=0 python ledwm/train.py --configs s1_train --jax.train_devices 0 --jax.policy_devices 0 --batch_size 40"
assert_not_contains "$output" "--xla_gpu_cuda_data_dir="
assert_not_contains "$output" "--batch_length 150"
assert_not_contains "$output" "--env.messenger.length 4"
assert_not_contains "$output" "--rssm.deter 256"
assert_not_contains "$output" "--task messenger_s1"
assert_not_contains "$output" "--load_exclude_key sent_embed"

output=$(DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0 \
  --batch-size 40 \
  --resume)

assert_contains "$output" "--resume"

output=$(DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0,2 \
  --batch-size 40 \
  --server opt \
  --preset s1_finetune_eval \
  -- --run.from_checkpoint '5711@02-23#22-01-45#585')

assert_contains "$output" "CUDA_VISIBLE_DEVICES=0,2"
assert_contains "$output" "RUN=TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 PYTHONHASHSEED=0 LEDWM_FAST_TRAIN_METRICS=1 LEDWM_FAST_OPTIMIZER_METRICS=1 LEDWM_SKIP_ADAM_METRICS=1 LEDWM_SKIP_TRAIN_OUTS=1 LOGURU_LEVEL=INFO LOGURU_ENQUEUE=1 CUDA_VISIBLE_DEVICES=0,2 python ledwm/train.py"
assert_contains "$output" "--configs s1_train s1_finetune_eval"
assert_contains "$output" "--jax.train_devices 0,1"
assert_contains "$output" "--jax.policy_devices 0"
assert_contains "$output" "--batch_size 40"
assert_contains "$output" "--run.server opt"
assert_contains "$output" "--run.from_checkpoint 5711@02-23#22-01-45#585"

output=$(DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0 \
  --batch-size 40 \
  --preset s1_finetune_eval)

assert_not_contains "$output" "--run.server"

output=$(DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0 \
  --batch-size 40 \
  --jax.mem_fraction 0.6)

assert_contains "$output" "XLA_PYTHON_CLIENT_MEM_FRACTION=0.6"
assert_contains "$output" "--jax.mem_fraction 0.6"

output=$(TF_ENABLE_ONEDNN_OPTS=1 TF_CPP_MIN_LOG_LEVEL=1 DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0 \
  --batch-size 40)

assert_contains "$output" "TF_CPP_MIN_LOG_LEVEL=1"
assert_contains "$output" "TF_ENABLE_ONEDNN_OPTS=1"

output=$(DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0 \
  --batch-size 40 \
  --log-level debug)

assert_contains "$output" "LOGURU_LEVEL=DEBUG"

output=$(LEDWM_FAST_TRAIN_METRICS=0 LEDWM_FAST_OPTIMIZER_METRICS=0 \
  LEDWM_SKIP_ADAM_METRICS=0 \
  LEDWM_SKIP_TRAIN_OUTS=0 DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0 \
  --batch-size 40)

assert_contains "$output" "LEDWM_FAST_TRAIN_METRICS=0"
assert_contains "$output" "LEDWM_FAST_OPTIMIZER_METRICS=0"
assert_contains "$output" "LEDWM_SKIP_ADAM_METRICS=0"
assert_contains "$output" "LEDWM_SKIP_TRAIN_OUTS=0"

output=$(DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0 \
  --batch-size 40 \
  --jax.prealloc False \
  --jax.allocator platform)

assert_contains "$output" "XLA_PYTHON_CLIENT_ALLOCATOR=platform"
assert_not_contains "$output" "--xla_gpu_cuda_data_dir="
assert_contains "$output" "--xla_gpu_autotune_level=0"
assert_contains "$output" "TF_CPP_MIN_LOG_LEVEL=2"
assert_contains "$output" "XLA_PYTHON_CLIENT_PREALLOCATE=false"

output=$(DRY_RUN=1 bash scripts/run_s1.sh \
  --gpus 0 \
  --batch-size 40 \
  --jax.prealloc False \
  --jax.allocator platform \
  --jax.xla_autotune_level 0 \
  --jax.quiet_xla True)

assert_contains "$output" "XLA_PYTHON_CLIENT_ALLOCATOR=platform"
assert_not_contains "$output" "--xla_gpu_cuda_data_dir="
assert_contains "$output" "--xla_gpu_autotune_level=0"
assert_contains "$output" "TF_CPP_MIN_LOG_LEVEL=2"
assert_contains "$output" "XLA_PYTHON_CLIENT_PREALLOCATE=false"
assert_contains "$output" "--jax.allocator platform"
assert_contains "$output" "--jax.xla_autotune_level 0"

rg -q '^s1_train:' ledwm/s1.yaml ||
  fail "expected ledwm/s1.yaml to define s1_train"

rg -q '^s1_finetune_eval:' ledwm/s1.yaml ||
  fail "expected ledwm/s1.yaml to define s1_finetune_eval"

if DRY_RUN=1 bash scripts/run_s1.sh --gpus 0,1,2 --batch-size 40 >/tmp/run_s1_test.out 2>&1; then
  fail "expected non-divisible batch size to fail"
fi
assert_contains "$(cat /tmp/run_s1_test.out)" "batch-size (40) is not divisible by the number of GPUs (3)"
