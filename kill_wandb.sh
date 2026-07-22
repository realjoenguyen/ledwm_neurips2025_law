#!/usr/bin/env bash
set -euo pipefail

username=${USER:?USER is not set}
pattern=wandb-service
pgrep -u "$username" -f "^$pattern" | while read -r pid; do
    echo "Killing process ID $pid"
    kill "$pid"
done
