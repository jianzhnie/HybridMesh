#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/jianzhnie/llmtuner/llm/TorchLLMTuner"
cd "$repo_root"

# Project-local Ascend environment. Override the model/run knobs below without
# editing the Python example.
if [[ "${LLMTUNER_SKIP_SET_ENV:-0}" != "1" ]]; then
  source "$repo_root/set_env.sh"
fi

export LLMTUNER_QWEN3_8B_PATH="${LLMTUNER_QWEN3_8B_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B}"
export LLMTUNER_DATASET_PATH="${LLMTUNER_DATASET_PATH:-/home/jianzhnie/llmtuner/hfhub/datasets/EleutherAI/hendrycks_math/train.jsonl}"
export LLMTUNER_GLOBAL_BATCH_SIZE="${LLMTUNER_GLOBAL_BATCH_SIZE:-8}"
export LLMTUNER_MAX_SEQ_LEN="${LLMTUNER_MAX_SEQ_LEN:-4096}"
export LLMTUNER_STEPS="${LLMTUNER_STEPS:-100}"
export LLMTUNER_DUMP_FOLDER="${LLMTUNER_DUMP_FOLDER:-$repo_root/outputs/qwen3-8b-npu}"
export LLMTUNER_SAVE_CHECKPOINT="${LLMTUNER_SAVE_CHECKPOINT:-1}"

mkdir -p "$LLMTUNER_DUMP_FOLDER/logs"
log_file="$LLMTUNER_DUMP_FOLDER/logs/train_$(date +%Y%m%d_%H%M%S).log"

nproc_per_node="${LLMTUNER_NPROC_PER_NODE:-8}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"
torchrun --master_addr="$master_addr" --master_port="$master_port" \
  --nproc_per_node="$nproc_per_node" \
  -m examples.train_qwen3_8b_npu 2>&1 | tee "$log_file"
