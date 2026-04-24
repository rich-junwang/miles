#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SRC_DIR=/opt/tiger/miles
DATA_DIR=/opt/tiger/dapo-math-17k
MEGATRON_PATH=/opt/tiger/megatron-lm
FULLY_ASYNC_DIR=${SRC_DIR}/examples/fully_async/
MODEL_DIR=/opt/tiger/models/qwen3_4b
MODEL_DIR_DIST=${MODEL_DIR}_torch_dist

# download model and data
hf download Qwen/Qwen3-4B --local-dir ${MODEL_DIR}
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir ${DATA_DIR}

#SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SRC_DIR}/scripts/models/qwen3-4B.sh"

PYTHONPATH=${MEGATRON_PATH} python ${SRC_DIR}/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${MODEL_DIR} \
    --save ${MODEL_DIR_DIST}

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --ref-load ${MODEL_DIR_DIST}
   --load ${MODEL_DIR}_miles
   --save ${MODEL_DIR}_miles/
   --save-interval 20
)


PROMPT_SET=${DATA_DIR}/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data

   # for staleness control
   #--max-weight-staleness 2
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${FULLY_ASYNC_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
