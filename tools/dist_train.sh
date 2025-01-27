#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
# PORT=${PORT:-43494}
# PORT=${PORT:-43498}
# PORT=${PORT:-43500}
PORT=${PORT:-43410}
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=2,3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py $CONFIG --launcher none ${@:3}


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --seed 1234

