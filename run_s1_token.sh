#!/bin/bash

# Assign command line arguments to variables
arg1=$1 # server
initial_arg2=$2 # batch_size
ids=$3 # This will replace the "1,2,3" part
configs=$4 # All configurations for --configs
other=$5

# Calculate the number of elements in ids
num_ids=$(echo $ids | tr -cd ',' | wc -c)
num_ids=$((num_ids + 1)) # Adding 1 because wc -c counts commas, not elements

# Check if $2 (initial_arg2) is divisible by num_ids
if [ $((initial_arg2 % num_ids)) -ne 0 ]; then
    echo -e "\033[31mERROR: batch_size ($initial_arg2) is not divisible by the number of IDs ($num_ids).\033[0m"
    exit 1
fi

# Calculate the decrement value as 5 times the number of elements in ids
decrement=$((5 * num_ids))

# Generate a sequence of numbers from 0 to num_ids-1
seq_ids=$(seq -s ',' 0 $((num_ids - 1)))

# Prepend or append specific strings to configs based on the first word
first_word=$(echo $configs | awk '{print $1;}')
# if [ "$first_word" = "sent" ]; then
#     configs="$configs"
# else
#     configs="reward_grain $configs "
# fi

echo "configs = $configs"

# Initial batch size
batch_size=$initial_arg2

# Initialize counter
count=0

# Maximum number of iterations
max_iterations=10
# if configs has the word overfit_eps then max_iteration = 1 instead
if [[ $configs == *"overfit_eps"* ]]; then
    max_iterations=1
fi

#! include s1_token at the end of $configs
configs="$configs s1_token"

# Loop for running commands with decrementing batch size
while [ $count -lt $max_iterations ]; do
    # Dynamic command with current batch size and configs
    cmd="sh scripts/run_messenger_s1.sh $ids --jax.train_devices $seq_ids --jax.policy_devices 0 --run.server $arg1 --configs $configs --batch_size $batch_size $other"
    
    if [ $count -eq 0 ]; then
        echo "batch_size = $batch_size"
    else
        echo -e "\033[31mFAIL. Command execution failed at batch_size = $batch_size. Exiting...\033[0m"
    fi
    
    # Print the command line in red
    echo -e "\033[31mExecuting command: $cmd\033[0m"
    
    # Execute the command
    eval "$cmd"
    
    # Update batch_size and count for the next iteration
    batch_size=$((batch_size - decrement))
    
    # assert batch_size > 0
    if [ $batch_size -le 0 ]; then
        echo -e "\033[31mERROR: batch_size is less than or equal to 0. Exiting...\033[0m"
        exit 1
    fi
    
    count=$((count + 1))
done
