#!/bin/bash

source /usr/local/apps/Anaconda/2021.11/etc/profile.d/conda.sh

conda activate unif_base
python CPDSSS.py &
pid1=$!

sleep 20

conda activate uniform2
python CPDSSS.py &
pid2=$!

cleanup() {
    echo "Terminating both scripts"
    kill $pid1 $pid2
    wait $pid1 $pid2
    echo "both scripts terminated"
}

trap 'cleanup' SIGINT

wait $pid1 $pid2