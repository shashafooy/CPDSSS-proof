#!/bin/bash

source /home/aaron/anaconda3/etc/profile.d/conda.sh

conda activate unif
python CPDSSS.py &

sleep 20

conda activate uniform2
python CPDSSS.py &

wait