#!/usr/bin/env bash

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

configname=$(basename -- "$CONFIG")
TASK_NAME="${configname%.*}"
mkdir -p ./search_exp/$TASK_NAME
TIME=$(date +"%Y%m%d_%H%M%S")

set -x

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/search.py \
    --cfg_path ${CONFIG} \
    --output_dir ./search_exp/$TASK_NAME \
    2>&1 | tee ./search_exp/$TASK_NAME/$TIME.log