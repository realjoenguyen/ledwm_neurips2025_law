#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_s2_test.sh --gpus IDS --checkpoint PATH [-- train flags...]

Options:
  --gpus IDS        CUDA device ids, for example 0 or 0,1,2,3.
  --checkpoint PATH Checkpoint used as the fine-tuning starting point.
  -h, --help        Show this help.

Example:
  scripts/run_s2_test.sh --gpus 0 --checkpoint ./checkpoint.ckpt
USAGE
}

die() {
    echo "ERROR: $*" >&2
    echo >&2
    usage >&2
    exit 1
}

gpu_ids=
checkpoint=
extra_train_args=()

while (($#)); do
    case "$1" in
        --gpus)
            [[ $# -ge 2 ]] || die "--gpus requires a value"
            gpu_ids=$2
            shift 2
            ;;
        --checkpoint)
            [[ $# -ge 2 ]] || die "--checkpoint requires a value"
            checkpoint=$2
            shift 2
            ;;
        --)
            shift
            extra_train_args+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            extra_train_args+=("$@")
            break
            ;;
        *)
            die "unexpected positional argument '$1'; use --gpus and --checkpoint"
            ;;
    esac
done

[[ -n $gpu_ids ]] || die "--gpus is required"
[[ -n $checkpoint ]] || die "--checkpoint is required"

# Define the full range for --critic_opt.lr with decimal notation
critic_lr_values=(0.00001 0.00002 0.00003 0.00004 0.00005 0.00006 0.00007 0.00008 0.00009 0.000001)

# Define the range for --step_finetune
step_finetune_values=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)

# Loop through each combination
for critic_lr in "${critic_lr_values[@]}"; do
    # Calculate the value for --actor_opt.lr as 2 * --critic_opt.lr using bc
    actor_lr=$(echo "$critic_lr * 2" | bc -l | tr -d '\n')

    # Debug output to ensure actor_lr is set correctly
    echo "Critic LR: $critic_lr, Actor LR: $actor_lr"

    if [ -z "$actor_lr" ]; then
        echo "Error: actor_opt.lr is empty"
        exit 1
    fi

    for step_finetune in "${step_finetune_values[@]}"; do
        configs=(
            sent
            large_encoder_s
            large_decoder_s
            small_image_data
            rew_smooth
            time
            two_cnn
            table
            multi_step
            no_image
            sum_reward
            prioritize
            no_decoder
            action_pred
            balanced_weight
            unimix_actor
            fewer_env_step
            small_rew_smooth
            finetune_policy
        )
        train_args=(
            --batch_length 150
            --replay.size 1e5
            --run.train_ratio 64
            --test_set test-se
            --env.messenger.length 32
            --imag_horizon 32
            --run.from_checkpoint "$checkpoint"
            --run.step_finetune "$step_finetune"
            --run.eps_finetune 1000
            --num_eval_eps 50
            --envs.amount 50
            --critic_opt.lr "$critic_lr"
            --actor_opt.lr "$actor_lr"
            "${extra_train_args[@]}"
        )
        command=(
            "$script_dir/run_s2.sh"
            --server dgx
            --batch-size 40
            --gpus "$gpu_ids"
            --configs "${configs[@]}"
            --preset s2
            -- "${train_args[@]}"
        )

        printf 'Running command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        "${command[@]}"
    done
done
