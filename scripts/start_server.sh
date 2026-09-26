#!/usr/bin/env bash

set -euo pipefail

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SERVED_NAME="llm-model"

HOST="0.0.0.0"
PORT=8000

MAX_MODEL_LEN=2000
GPU_MEMORY_UTILIZATION=0.85

MAX_NUM_SEQS=4
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
GPU_DEVICES="0"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --max-num-seqs N   Max concurrent sequences (default: 4)
  --tp N             Tensor parallel size (default: 1)
  --pp N             Pipeline parallel size (default: 1)
  --gpus LIST        CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1" (default: "0")
  -h, --help         Show this help

Examples:
  $0 --max-num-seqs 16
  $0 --max-num-seqs 8 --tp 2 --gpus 0,1
  $0 --max-num-seqs 8 --pp 2 --gpus 0,1
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-num-seqs)
            MAX_NUM_SEQS="$2"; shift 2 ;;
        --tp)
            TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --pp)
            PIPELINE_PARALLEL_SIZE="$2"; shift 2 ;;
        --gpus)
            GPU_DEVICES="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "ERROR: unknown argument: $1"
            usage ;;
    esac
done

if [[ ! "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --max-num-seqs must be a number"
    exit 1
fi

if [[ ! "$TENSOR_PARALLEL_SIZE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --tp must be a number"
    exit 1
fi

if [[ ! "$PIPELINE_PARALLEL_SIZE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --pp must be a number"
    exit 1
fi

if [[ ! "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: --gpus must be a comma-separated list of GPU indices, e.g. 0 or 0,1"
    exit 1
fi

NUM_GPUS_REQUESTED=$(( $(grep -o "," <<< "$GPU_DEVICES" | wc -l) + 1 ))
NUM_GPUS_NEEDED=$(( TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE ))

if [[ "$NUM_GPUS_REQUESTED" -ne "$NUM_GPUS_NEEDED" ]]; then
    echo "ERROR: --gpus lists $NUM_GPUS_REQUESTED device(s) but --tp $TENSOR_PARALLEL_SIZE x --pp $PIPELINE_PARALLEL_SIZE needs $NUM_GPUS_NEEDED"
    exit 1
fi

echo "Starting vLLM server..."
echo "Model:                   $MODEL"
echo "Served name:             $SERVED_NAME"
echo "max-num-seqs:            $MAX_NUM_SEQS"
echo "max-model-len:           $MAX_MODEL_LEN"
echo "gpu-memory-utilization:  $GPU_MEMORY_UTILIZATION"
echo "tensor-parallel-size:    $TENSOR_PARALLEL_SIZE"
echo "pipeline-parallel-size:  $PIPELINE_PARALLEL_SIZE"
echo "CUDA_VISIBLE_DEVICES:    $GPU_DEVICES"

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --dtype half