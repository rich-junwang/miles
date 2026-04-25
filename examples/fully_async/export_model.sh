#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
#pkill -9 python
sleep 3
pkill -9 ray
#pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16


SRC_DIR=/opt/tiger/miles
DATA_DIR=/opt/tiger/dapo-math-17k
MEGATRON_PATH=/opt/tiger/Megatron-LM
FULLY_ASYNC_DIR=${SRC_DIR}/examples/fully_async/
MODEL_DIR=/opt/tiger/models/qwen3_4b
MODEL_DIR_DIST=${MODEL_DIR}_torch_dist
NUM_GPUS=4

# download model and data
hf download Qwen/Qwen3-4B --local-dir ${MODEL_DIR}
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir ${DATA_DIR}

#SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SRC_DIR}/scripts/models/qwen3-4B.sh"

# single gpu
#PYTHONPATH=${MEGATRON_PATH}:${SRC_DIR} python ${SRC_DIR}/tools/convert_hf_to_torch_dist.py \
#    ${MODEL_ARGS[@]} \
#    --hf-checkpoint ${MODEL_DIR} \
#    --save ${MODEL_DIR_DIST}


# multiple gpus, we use multiple gpus when model size is too large
export LD_PRELOAD=/usr/local/src/nccl/build/lib/libnccl.so.2.29.3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NCHANNELS_PER_NET_PEER=4
export PYTHONPATH=${MEGATRON_PATH}:${SRC_DIR}
torchrun --nproc_per_node=4 --master_port=12356 \
${SRC_DIR}/tools/convert_hf_to_torch_dist.py \
${MODEL_ARGS[@]} \
--hf-checkpoint ${MODEL_DIR} \
--save ${MODEL_DIR_DIST}
echo "Model exporting is done!!"