#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
# PORT=${PORT:-29835}
# PORT=${PORT:-29838}
PORT=${PORT:-43500}
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test_cluster.py $CONFIG $CHECKPOINT --launcher none ${@:4}


