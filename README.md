# LLM Inference Benchmark

A reproducible benchmark project for evaluating **vLLM inference performance on a single GPU**.

The first experiment studies how changing `max-num-seqs` affects inference throughput and latency while keeping the model, GPU, workload, and other vLLM configuration parameters fixed.

---

## 1. Project Goal

The goal is to understand and measure the performance behavior of a production-style LLM inference server.

The initial experiment focuses on:

* vLLM inference serving
* GPU utilization
* GPU memory utilization
* request concurrency
* continuous batching
* throughput
* TTFT (Time To First Token)
* TPOT (Time Per Output Token)
* request queueing
* `max-num-seqs`

The project will first run as a **single-GPU inference benchmark**.

Kubernetes/EKS orchestration benchmarking will be added as a separate phase after the inference benchmark is established.

---

## 2. Current Benchmark

### Model

```text
Qwen/Qwen2.5-1.5B-Instruct
```

### Hardware

```text
GPU:       Tesla T4
GPU count: 1
```

### vLLM

```text
Version:                 0.30.0
Tensor Parallel Size:   1
Pipeline Parallel Size: 1
Max Model Length:       2000
GPU Memory Utilization: 0.85
```

---

## 3. Experiment Design

The experiment changes only one variable:

```text
max-num-seqs
```

Test values:

```text
4
8
12
16
```

All other benchmark parameters remain fixed.

### Fixed workload

```text
Dataset:          random
Input length:     430 tokens
Output length:    1000 tokens
Number of prompts: 50
Request rate:     2 requests/second
Repetitions:      3
Warm-up:          enabled
```

The benchmark currently uses vLLM's random dataset with controlled input and output lengths.

The `benchmark_prompt.txt` file is maintained separately and is not currently used as the request dataset by `vllm bench serve`.

---

## 4. Experiment Methodology

For each `max-num-seqs` value:

```text
1. Start a fresh vLLM server
2. Set max-num-seqs
3. Allow the server to initialize
4. Run the same benchmark workload
5. Save the raw benchmark result
6. Stop the server
7. Repeat for the next max-num-seqs value
```

Conceptually:

```text
max-num-seqs = 4
       ↓
benchmark
       ↓
result

max-num-seqs = 8
       ↓
benchmark
       ↓
result

max-num-seqs = 12
       ↓
benchmark
       ↓
result

max-num-seqs = 16
       ↓
benchmark
       ↓
result
```

The purpose is to isolate the effect of `max-num-seqs` rather than changing multiple variables at the same time.

---

## 5. Metrics

The benchmark results will be used to analyze metrics such as:

* Output token throughput
* Request throughput
* TTFT
* TPOT
* Request latency
* Queue time
* GPU utilization
* GPU memory utilization
* Error rate

The exact metric fields available in the generated vLLM benchmark result will be inspected after the first successful benchmark run.

No benchmark numbers are hard-coded into this repository.

---

## 6. Why `max-num-seqs`?

`max-num-seqs` controls the maximum number of sequences that vLLM can schedule concurrently.

Increasing it can allow the GPU to process more active sequences and may improve throughput when the GPU is underutilized.

However, increasing concurrency can also increase:

* KV-cache memory pressure
* GPU memory consumption
* queueing behavior
* latency
* contention between requests

Therefore, the goal is not simply to find the largest value.

The goal is to measure the relationship between:

```text
max-num-seqs
      ↓
concurrency
      ↓
batching
      ↓
GPU utilization
      ↓
throughput
      ↓
latency
```

---

## 7. Project Structure

```text
llm-inference-benchmark/
│
├── configs/
│   └── baseline.yaml
│
├── notebooks/
│   └── vllm-inference-benchmark.ipynb
│
├── plots/
│
├── results/
│   └── raw/
│
├── scripts/
│   ├── count_prompt_tokens.py
│   ├── run_benchmark.sh
│   └── start_server.sh
│
├── workloads/
│   ├── prompts/
│   │   └── benchmark_prompt.txt
│   └── workload.yaml
│
├── README.md
└── .gitignore
```

---

## 8. Scripts

### `scripts/start_server.sh`

Starts a vLLM server with a selected `max-num-seqs` value.

Example:

```bash
./scripts/start_server.sh 4
```

or:

```bash
./scripts/start_server.sh 16
```

The script keeps the remaining server configuration fixed.

---

### `scripts/run_benchmark.sh`

Runs the vLLM benchmark against the running server.

Example:

```bash
./scripts/run_benchmark.sh 4
```

The result is saved under:

```text
results/raw/
```

with a filename based on the tested `max-num-seqs` value.

---

### `scripts/count_prompt_tokens.py`

Loads the Qwen tokenizer and reports the character count and token count of:

```text
workloads/prompts/benchmark_prompt.txt
```

This is a utility for inspecting the prompt rather than the current random benchmark workload.

---

## 9. Configuration Files

### `configs/baseline.yaml`

Contains the baseline model, hardware, vLLM configuration, workload configuration, and experiment values.

### `workloads/workload.yaml`

Documents the benchmark workload:

```text
50 requests
2 RPS
430 input tokens
1000 output tokens
3 repetitions
warm-up enabled
```

---

## 10. Notebook

The main notebook is:

```text
notebooks/vllm-inference-benchmark.ipynb
```

The notebook is intended to provide the end-to-end benchmark workflow:

```text
GPU verification
      ↓
vLLM setup
      ↓
vLLM server startup
      ↓
health check
      ↓
benchmark execution
      ↓
raw JSON results
      ↓
metric extraction
      ↓
summary table
      ↓
plots
      ↓
analysis
```

The notebook is designed to run in a GPU environment such as Kaggle.

---

## 11. Execution Environment

Development is performed locally using VS Code.

The actual GPU benchmark is intended to run on Kaggle using a Tesla T4 GPU.

Development workflow:

```text
Local VS Code
      ↓
Project development
      ↓
GitHub
      ↓
Kaggle GPU environment
      ↓
Actual benchmark execution
      ↓
Raw results
      ↓
Analysis and plots
```

---

## 12. Reproducibility

The benchmark attempts to keep the following fixed across experiments:

```text
Model
GPU
vLLM version
Tensor parallelism
Pipeline parallelism
Max model length
GPU memory utilization
Input token length
Output token length
Request count
Request rate
Warm-up procedure
Benchmark repetitions
```

Only:

```text
max-num-seqs
```

is changed between the primary experiment runs.

---

## 13. Current Status

### Phase 1 — Project setup

* [x] Project structure created
* [x] Baseline configuration created
* [x] Workload configuration created
* [x] Benchmark scripts created
* [x] Notebook created
* [x] Git repository initialized
* [x] Initial commit created
* [x] GitHub repository created

### Phase 2 — Kaggle execution

* [ ] Upload/clone project into Kaggle
* [ ] Verify GPU
* [ ] Install/verify vLLM
* [ ] Start vLLM server
* [ ] Run benchmark
* [ ] Validate raw JSON result format

### Phase 3 — Benchmark experiment

* [ ] Run `max-num-seqs=4`
* [ ] Run `max-num-seqs=8`
* [ ] Run `max-num-seqs=12`
* [ ] Run `max-num-seqs=16`
* [ ] Repeat experiments
* [ ] Extract metrics
* [ ] Generate plots
* [ ] Analyze throughput/latency trade-offs

### Phase 4 — Infrastructure benchmark

After the single-GPU inference benchmark is complete:

```text
Kubernetes
    ↓
EKS
    ↓
GPU nodes
    ↓
vLLM Pods
    ↓
Scheduling
    ↓
Scaling
    ↓
Recovery
```

will be benchmarked separately.

---

## 14. Important Principle

This project separates two different questions:

### Inference benchmark

> How does vLLM perform on a GPU under a controlled workload?

### Orchestration benchmark

> How does Kubernetes/EKS manage and scale the inference workload?

They will be measured separately so that infrastructure effects do not get mixed with inference-engine performance.

---

## 15. Status

Current stage:

```text
Project setup complete
        ↓
GitHub repository ready
        ↓
Kaggle execution next
        ↓
Actual benchmark not yet executed
```

No benchmark performance numbers are reported until the workload is actually executed on the target GPU environment.
