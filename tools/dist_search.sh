#!/usr/bin/env bash

CFG_PATH=$1
GPUS=$2
PORT=${PORT:-29500}
TIME=$(date +"%Y%m%d_%H%M%S")

configname=$(basename -- "$CFG_PATH")
TASK_NAME="${configname%.*}"
mkdir -p ./search_exp/$TASK_NAME

PYTHONWARNINGS='ignore' \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/search.py --cfg_path $CFG_PATH \
    --output_dir ./search_exp/$TASK_NAME \
    2>&1 | tee ./search_exp/$TASK_NAME/$TIME.log ${@:3}
