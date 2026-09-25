#!/usr/bin/env bash

set -euo pipefail

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SERVED_NAME="llm-model"

HOST="0.0.0.0"
PORT=8000

MAX_MODEL_LEN=2000
GPU_MEMORY_UTILIZATION=0.85

TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1

MAX_NUM_SEQS="${1:-4}"

if [[ ! "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: max-num-seqs must be a number"
    exit 1
fi

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Served name: $SERVED_NAME"
echo "max-num-seqs: $MAX_NUM_SEQS"
echo "max-model-len: $MAX_MODEL_LEN"
echo "gpu-memory-utilization: $GPU_MEMORY_UTILIZATION"

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --dtype half