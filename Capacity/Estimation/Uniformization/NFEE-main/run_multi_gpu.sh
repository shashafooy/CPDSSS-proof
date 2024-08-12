#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <python_script> <N_instances>"
  exit 1
fi

# Assign the first argument to a variable
PYTHON_SCRIPT=$1
N_RUNS=$2

# Check if the file exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: File $PYTHON_SCRIPT not found!"
  exit 1
fi

CONDA_BASE=$(conda info --base)

source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate gpuknn

PIDS=()
for ((i=1; i<=$N_RUNS; i++)); do
	echo "Starting instance $i"
	python "$PYTHON_SCRIPT" &
	PIDS+=($!)
done

cleanup() {
    for pid in "${PIDS[@]}"; do
	echo "Killing process $pid"
	kill "$pid"
	wait "$pid"
    done
    echo "Scripts terminated"
}

trap 'cleanup' SIGINT

wait
