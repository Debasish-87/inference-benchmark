#!/usr/bin/env bash

set -euo pipefail

BASE_URL="http://127.0.0.1:8000"

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SERVED_NAME="llm-model"

DATASET="random"

INPUT_LEN=430
OUTPUT_LEN=1000

NUM_PROMPTS=50

RESULT_DIR="results/raw"

MAX_NUM_SEQS=4
REQUEST_RATE=2
RUN_ID=""

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --max-num-seqs N     Max concurrent sequences the server was started with (default: 4)
  --request-rate N     Requests per second to offer, e.g. 0.2, 1, 2 (default: 2)
  --run-id ID          Label for this run, e.g. warmup, 1, 2, 3 (default: none)
  -h, --help           Show this help

Examples:
  $0 --max-num-seqs 16 --request-rate 0.2
  $0 --max-num-seqs 8  --request-rate 0.5 --run-id 1
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-num-seqs)
            MAX_NUM_SEQS="$2"; shift 2 ;;
        --request-rate)
            REQUEST_RATE="$2"; shift 2 ;;
        --run-id)
            RUN_ID="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "ERROR: unknown argument: $1"
            usage ;;
    esac
done

if [[ ! "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --max-num-seqs must be a whole number"
    exit 1
fi

if [[ ! "$REQUEST_RATE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: --request-rate must be a positive number (e.g. 0.2, 1, 2)"
    exit 1
fi

RATE_LABEL="${REQUEST_RATE//./p}"
RESULT_FILE="max-num-seqs-${MAX_NUM_SEQS}-rate-${RATE_LABEL}"
if [[ -n "$RUN_ID" ]]; then
    RESULT_FILE="${RESULT_FILE}-run-${RUN_ID}"
fi
RESULT_FILE="${RESULT_FILE}.json"

mkdir -p "$RESULT_DIR"

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
echo "Run ID:           ${RUN_ID:-(none)}"

echo "Result:           $RESULT_DIR/$RESULT_FILE"

echo "=========================================="

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