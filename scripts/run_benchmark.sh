#!/usr/bin/env bash

set -euo pipefail

# --------------------------------------------------
# Fixed benchmark configuration
# --------------------------------------------------

BASE_URL="http://127.0.0.1:8000"

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SERVED_NAME="llm-model"

DATASET="random"

INPUT_LEN=430
OUTPUT_LEN=1000

NUM_PROMPTS=50
REQUEST_RATE=2

RESULT_DIR="results/raw"

# Independent variable
MAX_NUM_SEQS="${1:-4}"

RESULT_FILE="max-num-seqs-${MAX_NUM_SEQS}.json"


# --------------------------------------------------
# Validation
# --------------------------------------------------

if [[ ! "$MAX_NUM_SEQS" =~ ^(4|8|12|16)$ ]]; then
    echo "ERROR: max-num-seqs must be one of: 4, 8, 12, 16"
    exit 1
fi


# --------------------------------------------------
# Result directory
# --------------------------------------------------

mkdir -p "$RESULT_DIR"


# --------------------------------------------------
# Benchmark information
# --------------------------------------------------

echo "=========================================="
echo "vLLM Inference Benchmark"
echo "=========================================="

echo "Model:            $MODEL"
echo "Served name:      $SERVED_NAME"
echo "Dataset:          $DATASET"

echo "max-num-seqs:     $MAX_NUM_SEQS"

echo "Input tokens:     $INPUT_LEN"
echo "Output tokens:    $OUTPUT_LEN"

echo "Requests:         $NUM_PROMPTS"
echo "Request rate:     $REQUEST_RATE RPS"

echo "Result:           $RESULT_DIR/$RESULT_FILE"

echo "=========================================="


# --------------------------------------------------
# Run benchmark
# --------------------------------------------------

vllm bench serve \
    --backend vllm \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --dataset-name "$DATASET" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$RESULT_FILE"


echo
echo "=========================================="
echo "Benchmark completed"
echo "=========================================="
echo "Result saved to:"
echo "$RESULT_DIR/$RESULT_FILE"