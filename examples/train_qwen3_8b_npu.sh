#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/jianzhnie/llmtuner/llm/HybridMesh"
cd "$repo_root"

# Project-local Ascend environment. Override the model/run knobs below without
# editing the Python example.
if [[ "${HPMESH_SKIP_SET_ENV:-0}" != "1" ]]; then
  source "$repo_root/set_env.sh"
fi

export HPMESH_QWEN3_8B_PATH="${HPMESH_QWEN3_8B_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B}"
export HPMESH_GLOBAL_BATCH_SIZE="${HPMESH_GLOBAL_BATCH_SIZE:-8}"
export HPMESH_MAX_SEQ_LEN="${HPMESH_MAX_SEQ_LEN:-2048}"
export HPMESH_STEPS="${HPMESH_STEPS:-20}"
export HPMESH_DUMP_FOLDER="${HPMESH_DUMP_FOLDER:-$repo_root/outputs/qwen3-8b-npu}"

nproc_per_node="${HPMESH_NPROC_PER_NODE:-8}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"
torchrun --master_addr="$master_addr" --master_port="$master_port" \
  --nproc_per_node="$nproc_per_node" \
  -m examples.train_qwen3_8b_npu
