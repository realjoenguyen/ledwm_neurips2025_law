#!/bin/bash

# Assign command line arguments to variables
arg1=$1 # server
initial_arg2=$2 # batch size
ids=$3 # This will replace the "1,2,3" part

# Calculate the number of elements in ids
num_ids=$(echo $ids | tr -cd ',' | wc -c)
num_ids=$((num_ids + 1)) # Adding 1 because wc -c counts commas, not elements

# Check if $2 is divisible by num_ids
if [ $((initial_arg2 % num_ids)) -ne 0 ]; then
    echo -e "\033[31mERROR: batch_size ($initial_arg2) is not divisible by the number of IDs ($num_ids).\033[0m"
    exit 1
fi

# Calculate the decrement value as 5 times the number of elements in ids
decrement=$((5 * num_ids))

# Generate a sequence of numbers from 0 to num_ids-1
seq_ids=$(seq -s ',' 0 $((num_ids - 1)))

# Initial batch size
batch_size=$initial_arg2

# Initialize counter
count=0

# Maximum number of iterations
max_iterations=10

# Loop for running commands with decrementing batch size
while [ $count -lt $max_iterations ]; do
    # Dynamic command with current batch size
    cmd="sh scripts/run_messenger_s1.sh $ids --jax.train_devices $seq_ids --jax.policy_devices 0 --run.server $arg1 --configs sent large_encoder large_decoder reward_grain sent_cache no_atten dense_image long_read random_policy --envs.amount 50 --batch_size $batch_size"
    
    if [ $count -eq 0 ]; then
        echo "batch_size = $batch_size"
    else
        echo -e "\033[31mFAIL!. batch_size = $batch_size\n\033[0m"
    fi
    
    echo -e "\033[32mExecuting command: $cmd\n\033[0m"
    
    # Execute the command
    eval "$cmd"
    
    # Update batch_size and count for the next iteration
    batch_size=$((batch_size - decrement))
    count=$((count + 1))
done
