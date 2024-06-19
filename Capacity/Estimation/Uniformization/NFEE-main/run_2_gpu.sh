#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <python_script>"
  exit 1
fi

# Assign the first argument to a variable
PYTHON_SCRIPT=$1

# Check if the file exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: File $PYTHON_SCRIPT not found!"
  exit 1
fi

CONDA_BASE=$(conda info --base)

source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate unif
python "$PYTHON_SCRIPT" &
pid1=$!

sleep 10

python "$PYTHON_SCRIPT" &
pid2=$!

cleanup() {
    echo "Terminating both scripts"
    kill $pid1 $pid2
    wait $pid1 $pid2
    echo "both scripts terminated"
}

trap 'cleanup' SIGINT

wait $pid1 $pid2
