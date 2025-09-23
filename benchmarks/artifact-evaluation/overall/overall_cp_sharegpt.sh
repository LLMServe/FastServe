#!/bin/bash

# --- 1. Conda Environment Activation ---
# This ensures the script runs within the correct conda environment.
# Find the base conda installation and source its profile script.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fastserve-rr-vllm

# --- 2. Define Backend and Frontend Parameters ---
# Backend (vLLM) specific parameters
BACKEND_HOST="0.0.0.0"
BACKEND_PORT="7845"
MODEL_PATH="/users/zzl/models/opt-13b"
TOKENIZER_PATH="/users/wby/weights/opt-13b" # Note: Adjusted to match your frontend command
DATASET_PATH="/users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
PROCESS_NAME="possion"
DATASET_NAME="sharegpt"

LOG_DIR="/users/zzl/FastServe/benchmarks/artifact-evaluation/overall/logs"

# Create log directory if it doesn't exist.
mkdir -p "$LOG_DIR"

# Ensure the script is running from the correct directory.
echo "Changing directory to FastServe..."
cd ~/FastServe

# --- 3. Define and run the test loop ---
# num_prompt_list=(20 50 100 200 300 350 400 400 400 400 400)
# rate_list=(0.1 0.2 0.5 1 1.5 1.75 2 2.25 2.5 3 4)
num_prompt_list=(100 200 300 350 400 400 400 400 400)
rate_list=(0.5 1 1.5 1.75 2 2.25 2.5 3 4)
num_tests=${#rate_list[@]}

for (( i=0; i<num_tests; i++ )); do
    num_prompt=${num_prompt_list[i]}
    rate=${rate_list[i]}
    
    # Define the log file name for the current test.
    LOG_FILE="${LOG_DIR}/exp-cp-${DATASET_NAME}-${num_prompt}-${rate}-client-log"

    echo "======================================================================"
    echo "Running Test $((i+1))/$num_tests: Rate = $rate, Num Prompts = $num_prompt"
    echo "Logging frontend output to: $LOG_FILE"
    echo "======================================================================"

    ## Start vLLM Backend
    echo "Starting vLLM backend..."
    # The command is run in the background (&) and its output is discarded for a clean terminal.
    python -m vllm.entrypoints.api_server \
        --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
        --model "$MODEL_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --load-format dummy --dtype float16 \
        --block-size 16 --gpu-memory-utilization 0.9 \
        --max-num-batched-tokens 2048 \
        --max-model-len 2048 --max-num-seqs 16 \
        --enforce-eager --disable-custom-all-reduce \
        --device cuda --num-scheduler-steps 1 \
        --disable-log-requests --num-gpu-blocks-override 400 \
        --enable-chunked-prefill &

    # Store the Process ID (PID) of the background process.
    BACKEND_PID=$!
    echo "vLLM backend started with PID: $BACKEND_PID"

    # Wait for the backend to start up properly.
    echo "Waiting for backend to warm up..."
    sleep 30 # Increased sleep time for vLLM to load the model.

    ## Run Frontend Benchmark with Logging
    echo "Starting vLLM frontend benchmark..."
    # The `2>&1 | tee` command captures both standard output and standard error and redirects them to both the console and the log file.
    python benchmarks/benchmark-serving/benchmark-serving.py \
        --port "$BACKEND_PORT" \
        --backend vllm \
        --tokenizer "$TOKENIZER_PATH" \
        --dataset "$DATASET_PATH" \
        --num-prompts "$num_prompt" \
        --request-rate "$rate" \
        --process-name "$PROCESS_NAME" \
        --dataset-name "$DATASET_NAME" \
        2>&1 | tee "$LOG_FILE"

    ## Clean up
    echo "Benchmark finished. Terminating backend process..."
    # Check if the process is still running and then kill it.
    if ps -p $BACKEND_PID > /dev/null
    then
        kill -9 $BACKEND_PID
        echo "Backend process $BACKEND_PID terminated."
    else
        echo "Backend process $BACKEND_PID was already terminated."
    fi
    echo "======================================================================"
    echo ""
    
    ## Post-test Cooldown
    echo "Waiting for 30 seconds before next test..."
    sleep 30
done

echo "All tests are completed."