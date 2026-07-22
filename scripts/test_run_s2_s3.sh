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

mock_bin=$(mktemp -d)
trap 'rm -rf "$mock_bin" /tmp/run_s2_s3_test.out' EXIT

cat >"$mock_bin/python" <<'PYTHON'
#!/usr/bin/env bash
batch_size=
compile_only=0
while (($#)); do
  case "$1" in
    --batch_size)
      batch_size=$2
      shift 2
      ;;
    --run.compile_only)
      [[ $2 == true ]] && compile_only=1
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -n ${LEDWM_TEST_AUTOBATCH_MAX_FIT:-} ]]; then
  if ((compile_only)); then
    printf '%s\n' "$batch_size" >>"$LEDWM_TEST_AUTOBATCH_TRACE"
    if [[ -n ${LEDWM_TEST_AUTOBATCH_POLICY_TRACE:-} ]]; then
      printf '%s\n' "${LEDWM_CAPACITY_PROBE_POLICY:-unset}" \
        >>"$LEDWM_TEST_AUTOBATCH_POLICY_TRACE"
    fi
    ((batch_size <= LEDWM_TEST_AUTOBATCH_MAX_FIT))
    exit
  fi
  exit 0
fi
echo "mock python should not run for dry-run wrapper tests" >&2
exit 42
PYTHON
chmod +x "$mock_bin/python"

cat >"$mock_bin/nvidia-smi" <<'NVIDIA_SMI'
#!/usr/bin/env bash
case "$*" in
  *--query-gpu=compute_cap*)
    echo "9.0"
    ;;
  *--query-gpu=memory.used*)
    echo "0"
    ;;
  *--query-gpu=uuid,name,memory.total*)
    case "$*" in
      *"-i 0 "*) echo "GPU-mock-0, Mock GPU, 81920" ;;
      *"-i 1 "*) echo "GPU-mock-1, Mock GPU, 81920" ;;
      *) exit 1 ;;
    esac
    ;;
  *--query-gpu=name,memory.total*)
    echo "Mock GPU, 81920"
    ;;
  *)
    exit 1
    ;;
esac
NVIDIA_SMI
chmod +x "$mock_bin/nvidia-smi"

cat >"$mock_bin/sha256sum" <<'SHA256SUM'
#!/usr/bin/env bash
input=
if (($# == 0)); then
  input=$(cat)
fi
if [[ -n ${LEDWM_TEST_AUTOBATCH_HASH_INPUT_TRACE:-} && $input == s3$'\n'* ]]; then
  printf '%s' "$input" >"$LEDWM_TEST_AUTOBATCH_HASH_INPUT_TRACE"
fi
case "$input" in
  *GPU-mock-0*) echo "autobatch-gpu-0  -" ;;
  *GPU-mock-1*) echo "autobatch-gpu-1  -" ;;
  *) echo "autobatch-test  -" ;;
esac
SHA256SUM
chmod +x "$mock_bin/sha256sum"

mkdir -p "$mock_bin/autobatch-cache"
printf '%s\n' 512 >"$mock_bin/autobatch-cache/autobatch-test.max_batch"
printf '%s\n' 512 >"$mock_bin/autobatch-cache/autobatch-gpu-0.max_batch"
export PATH="$mock_bin:$PATH"

search_cache="$mock_bin/autobatch-search-cache"
search_trace="$mock_bin/autobatch-search-trace"
mkdir -p "$search_cache"
output=$(
  LEDWM_AUTOBATCH_CACHE_DIR="$search_cache" \
  LEDWM_TEST_AUTOBATCH_MAX_FIT=640 \
  LEDWM_TEST_AUTOBATCH_TRACE="$search_trace" \
  bash scripts/run_s2.sh \
    --gpus 0 \
    --batch-size auto \
    --auto-batch-min 128 \
    --auto-batch-max 1024 \
    --auto-batch-quantum 128 \
    --auto-batch-safety 100 2>&1
)
[[ $(tr '\n' ' ' <"$search_trace") == "1024 512 768 640 " ]] ||
  fail "expected max-first binary probes: 1024 512 768 640"
assert_contains "$output" "autobatch.cache_written | max_fit=640"
assert_contains "$output" "autobatch.selected | batch_size=640 | max_fit=640 | safety=100%"
[[ $(<"$search_cache/autobatch-gpu-0.max_batch") == 640 ]] ||
  fail "expected max-first search to cache max_fit=640"

s3_search_cache="$mock_bin/s3-autobatch-search-cache"
s3_search_trace="$mock_bin/s3-autobatch-search-trace"
s3_policy_trace="$mock_bin/s3-autobatch-policy-trace"
mkdir -p "$s3_search_cache"
output=$(
  LEDWM_AUTOBATCH_CACHE_DIR="$s3_search_cache" \
  LEDWM_TEST_AUTOBATCH_MAX_FIT=384 \
  LEDWM_TEST_AUTOBATCH_TRACE="$s3_search_trace" \
  LEDWM_TEST_AUTOBATCH_POLICY_TRACE="$s3_policy_trace" \
  bash scripts/run_s3.sh \
    --gpus 0 \
    --batch-size auto \
    --auto-batch-min 128 \
    --auto-batch-max 512 \
    --auto-batch-quantum 128 \
    --auto-batch-safety 100 2>&1
)
[[ $(tr '\n' ' ' <"$s3_search_trace") == "128 256 512 384 " ]] ||
  fail "expected S3 exponential and binary probes: 128 256 512 384"
[[ $(tr '\n' ' ' <"$s3_policy_trace") == "1 0 0 0 " ]] ||
  fail "expected S3 policy warmup only on the first capacity probe"
assert_contains "$output" "autobatch.cache_written | max_fit=384"

hash_input_trace="$mock_bin/s3-autobatch-hash-input"
output=$(LEDWM_AUTOBATCH_CACHE_DIR="$mock_bin/autobatch-cache" \
  LEDWM_TEST_AUTOBATCH_HASH_INPUT_TRACE="$hash_input_trace" DRY_RUN=1 \
  bash scripts/run_s3.sh \
    --gpus 0 \
    --batch-size auto \
    --auto-batch-min 128 \
    --auto-batch-max 512 \
    --auto-batch-quantum 128 \
    --envs.amount 100 \
    --replay.size 2e4 \
    --replay.min_size 64 \
    --run.train_ratio 128 \
    --batch_length 150 2>&1)
assert_contains "$output" "autobatch.cache_hit | max_fit=512"
hash_input=$(<"$hash_input_trace")
assert_not_contains "$hash_input" "--envs.amount"
assert_not_contains "$hash_input" "--replay.size"
assert_not_contains "$hash_input" "--replay.min_size"
assert_not_contains "$hash_input" "--run.train_ratio"
assert_contains "$hash_input" "--batch_length 150"

check_wrapper() {
  local script=$1
  local launcher=$2
  local default_config=$3
  local preset=$4
  local launch_command

  local output
  output=$(DRY_RUN=1 bash "$script" \
    --gpus 0 \
    --batch-size 40)

  assert_contains "$output" "--configs $default_config"
  assert_contains "$output" "CUDA_VISIBLE_DEVICES=0"
  assert_contains "$output" "LOGURU_LEVEL=INFO"
  assert_contains "$output" "LOGURU_ENQUEUE=1"
  if [[ $launcher == direct ]]; then
    launch_command="python ledwm/train.py"
  else
    launch_command="bash $launcher 0"
  fi
  assert_contains "$output" "RUN=LOGURU_LEVEL=INFO LOGURU_ENQUEUE=1 CUDA_VISIBLE_DEVICES=0 $launch_command"
  assert_contains "$output" "--jax.train_devices 0"
  assert_contains "$output" "--jax.policy_devices 0"
  assert_contains "$output" "--batch_size 40"
  assert_not_contains "$output" "--run.server"

  output=$(LEDWM_AUTOBATCH_CACHE_DIR="$mock_bin/autobatch-cache" DRY_RUN=1 \
    bash "$script" \
    --gpus 0 \
    --batch-size auto \
    --auto-batch-min 128 \
    --auto-batch-max 512 \
    --auto-batch-quantum 128 \
    --auto-batch-safety 75 \
    --envs.amount 200 \
    --batch_length 60 \
    --use_wandb False 2>&1)

  assert_contains "$output" "autobatch.cache_hit | max_fit=512"
  assert_contains "$output" "autobatch.selected | batch_size=384 | max_fit=512 | safety=75%"
  assert_contains "$output" "--batch_size 384"
  assert_not_contains "$output" "--batch_size auto"
  assert_contains "$output" "--envs.amount 200"
  assert_contains "$output" "--batch_length 60"
  assert_contains "$output" "--use_wandb False"

  if LEDWM_AUTOBATCH_CACHE_DIR="$mock_bin/autobatch-cache" DRY_RUN=1 \
    bash "$script" \
    --gpus 1 \
    --batch-size auto \
    --auto-batch-min 128 \
    --auto-batch-max 512 \
    --auto-batch-quantum 128 >/tmp/run_s2_s3_test.out 2>&1; then
    if [[ $script == scripts/run_s2.sh ]]; then
      fail "expected a different physical GPU to miss the S2 auto-batch cache"
    fi
  elif [[ $script != scripts/run_s2.sh ]]; then
    fail "expected the unchanged S3 model-level auto-batch cache to hit"
  fi
  if [[ $script == scripts/run_s2.sh ]]; then
    assert_contains "$(cat /tmp/run_s2_s3_test.out)" \
      "auto batch cache miss during --dry-run"
  fi

  output=$(DRY_RUN=1 bash "$script" \
    --gpus 0 \
    --batch-size 40 \
    --resume)

  assert_contains "$output" "--resume"

  output=$(DRY_RUN=1 bash "$script" \
    --gpus 0,2 \
    --batch-size 40 \
    --server dgx \
    --preset "$preset" \
    -- --run.from_checkpoint '5711@02-23#22-01-45#585')

  assert_contains "$output" "CUDA_VISIBLE_DEVICES=0,2"
  if [[ $launcher == direct ]]; then
    launch_command="python ledwm/train.py"
  else
    launch_command="bash $launcher 0,2"
  fi
  assert_contains "$output" "RUN=LOGURU_LEVEL=INFO LOGURU_ENQUEUE=1 CUDA_VISIBLE_DEVICES=0,2 $launch_command"
  assert_contains "$output" "--configs $default_config $preset"
  assert_contains "$output" "--jax.train_devices 0,1"
  assert_contains "$output" "--jax.policy_devices 0"
  assert_contains "$output" "--batch_size 40"
  assert_contains "$output" "--run.server dgx"
  assert_contains "$output" "--run.from_checkpoint 5711@02-23#22-01-45#585"

  output=$(DRY_RUN=1 bash "$script" \
    --gpus 0 \
    --batch-size 40 \
    --configs sent large_encoder \
    --jax.mem_fraction 0.6)

  assert_contains "$output" "--configs sent large_encoder"
  assert_not_contains "$output" "--configs $default_config"
  assert_contains "$output" "XLA_PYTHON_CLIENT_MEM_FRACTION=0.6"
  assert_contains "$output" "--jax.mem_fraction 0.6"

  output=$(DRY_RUN=1 bash "$script" \
    --gpus 0 \
    --batch-size 40 \
    --log-level warning)

  assert_contains "$output" "LOGURU_LEVEL=WARNING"

  output=$(DRY_RUN=1 bash "$script" \
    --gpus 0 \
    --batch-size 40 \
    --jax.prealloc False \
    --jax.allocator platform)

  assert_contains "$output" "XLA_PYTHON_CLIENT_ALLOCATOR=platform"
  assert_not_contains "$output" "--xla_gpu_cuda_data_dir="
  assert_contains "$output" "--xla_gpu_autotune_level=0"
  assert_contains "$output" "TF_CPP_MIN_LOG_LEVEL=2"
  assert_contains "$output" "XLA_PYTHON_CLIENT_PREALLOCATE=false"
  assert_contains "$output" "--jax.allocator platform"

  if DRY_RUN=1 bash "$script" --gpus 0,1,2 --batch-size 40 >/tmp/run_s2_s3_test.out 2>&1; then
    fail "expected non-divisible batch size to fail for $script"
  fi
  assert_contains "$(cat /tmp/run_s2_s3_test.out)" "batch-size (40) is not divisible by the number of GPUs (3)"
}

check_wrapper scripts/run_s2.sh direct "s2_train" s2_token
check_wrapper scripts/run_s3.sh direct "s3_train" small_s3

output=$(DRY_RUN=1 bash scripts/run_lwm.sh \
  --task hard \
  --gpus 0 \
  --batch-size 40 \
  --envs.amount 24)
assert_contains "$output" "--configs lwm_train"
assert_contains "$output" "--batch_size 40"
assert_contains "$output" "--task lwm_hard"
assert_contains "$output" "--rssm.task hard"
assert_contains "$output" "--envs.amount 24"

output=$(DRY_RUN=1 bash scripts/run_lwm.sh \
  --gpus 0,1 \
  --batch_size 200 \
  --envs.amount 300 \
  --batch_length 64 \
  --replay.size 2e4 \
  --run.train_ratio 128 \
  --use_wandb False \
  --task easy)
assert_contains "$output" "CUDA_VISIBLE_DEVICES=0,1"
assert_contains "$output" "--batch_size 200"
assert_contains "$output" "--task lwm_easy"
assert_contains "$output" "--rssm.task easy"
assert_contains "$output" "--envs.amount 300"
assert_contains "$output" "--batch_length 64"
assert_contains "$output" "--replay.size 2e4"
assert_contains "$output" "--run.train_ratio 128"
assert_contains "$output" "--use_wandb False"

output=$(LEDWM_AUTOBATCH_CACHE_DIR="$mock_bin/autobatch-cache" DRY_RUN=1 \
  bash scripts/run_lwm.sh \
  --task medium \
  --gpus 0 \
  --batch-size auto \
  --auto-batch-min 128 \
  --auto-batch-max 512 \
  --auto-batch-quantum 128 \
  --auto-batch-safety 75 2>&1)
assert_contains "$output" "autobatch.cache_hit | max_fit=512"
assert_contains "$output" "autobatch.selected | batch_size=384 | max_fit=512 | safety=75%"
assert_contains "$output" "--batch_size 384"
assert_contains "$output" "--task lwm_medium"

output=$(DRY_RUN=1 bash scripts/run_lwm.sh \
  --task easy \
  --gpus 0 \
  --batch-size 40 \
  --configs sent large_encoder)
assert_contains "$output" "--configs sent large_encoder"
assert_not_contains "$output" "--configs lwm_train"

if DRY_RUN=1 bash scripts/run_lwm.sh \
  --task impossible --gpus 0 --batch-size 40 >/tmp/run_s2_s3_test.out 2>&1; then
  fail "expected invalid Messenger-LWM task to fail"
fi
assert_contains "$(cat /tmp/run_s2_s3_test.out)" "--task must be easy, medium, or hard"

bash -n run_finetune.sh scripts/run_s2_test.sh ||
  fail "expected run_s2 caller scripts to pass shell syntax checks"

if rg -n '\./run_s2\.sh dgx 40' run_finetune.sh scripts/run_s2_test.sh; then
  fail "expected run_s2 caller scripts to use the flag-based wrapper interface"
fi

rg -q '^s2:' ledwm/s2.yaml ||
  fail "expected ledwm/s2.yaml to define s2"

rg -q '^s2_train:' ledwm/s2.yaml ||
  fail "expected ledwm/s2.yaml to define s2_train"

if rg -q -- '--imag_horizon' scripts/run_s2.sh; then
  fail "expected S2 launcher to use the s2_train imag_horizon default"
fi

[[ ! -e scripts/run_messenger_s2.sh ]] ||
  fail "expected S2 child launcher to be merged into scripts/run_s2.sh"

[[ ! -e scripts/run_messenger_s3.sh ]] ||
  fail "expected S3 child launcher to be merged into scripts/run_s3.sh"

rg -q '^s2_token:' ledwm/s2.yaml ||
  fail "expected ledwm/s2.yaml to define s2_token"

rg -q '^s3:' ledwm/s3.yaml ||
  fail "expected ledwm/s3.yaml to define s3"

rg -q '^s3_train:' ledwm/s3.yaml ||
  fail "expected ledwm/s3.yaml to define s3_train"

rg -q '^small_s3:' ledwm/s3.yaml ||
  fail "expected ledwm/s3.yaml to define small_s3"

rg -q '^lwm:' ledwm/lwm.yaml ||
  fail "expected ledwm/lwm.yaml to define lwm"

rg -q '^lwm_train:' ledwm/lwm.yaml ||
  fail "expected ledwm/lwm.yaml to define lwm_train"

rg -q '^lwm_small:' ledwm/lwm.yaml ||
  fail "expected ledwm/lwm.yaml to define lwm_small"
