#!/bin/bash

source ~/.bashrc
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fastserve

# Define backend parameters
BACKEND_HOST="0.0.0.0"
BACKEND_PORT="10002"
MODEL_PATH="/users/lsy/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/"
BLOCK_SIZE="16"
GPU_UTIL="0.9"
SWAP_SPACE="1"
SCHED_POLICY="fcfs"
MAX_BATCH_SIZE="16"
MAX_TOKENS_PER_BATCH="2048"

# Define frontend parameters
TOKENIZER_PATH="/users/lsy/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/"
DATASET_PATH="/users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
PROCESS_NAME="possion"
DATASET_NAME="sharegpt"
LOG_DIR="/users/zzl/FastServe/benchmarks/artifact-evaluation/gqa/logs"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"
cd ~/FastServe

# Define the lists of parameters as shell arrays
num_prompt_list=(200 400 600 660 700 760 800 800 1000)
rate_list=(1 2 3 3.3 3.5 3.8 4 5 5.5)

# Get the total number of elements in the array
num_tests=${#rate_list[@]}

# Loop from 0 to the number of tests - 1
for (( i=0; i<num_tests; i++ )); do
    # Get the parameters for the current run
    num_prompt=${num_prompt_list[i]}
    rate=${rate_list[i]}

    # Define the log file name for the current test
    LOG_FILE="${LOG_DIR}/exp-fcfs-${DATASET_NAME}-${num_prompt}-${rate}-client-log"

    # Print a message indicating which test is running
    echo "======================================================================"
    echo "Running Test $((i+1))/$num_tests: Rate = $rate, Num Prompts = $num_prompt"
    echo "Logging frontend output to: $LOG_FILE"
    echo "======================================================================"

    # --- Start Backend ---
    echo "Starting FastGen backend..."
    python3 fastserve/api_server/fastserve_api_server.py \
        --host "$BACKEND_HOST" \
        --port "$BACKEND_PORT" \
        --model "$MODEL_PATH" \
        --block-size "$BLOCK_SIZE" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --swap-space "$SWAP_SPACE" \
        --sched-policy "$SCHED_POLICY" \
        --max-batch-size "$MAX_BATCH_SIZE" \
        --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH" &

    BACKEND_PID=$!
    echo "FastGen backend started with PID: $BACKEND_PID"
    sleep 30

    # --- Start Frontend with Tee to log file, capturing both stdout and stderr ---
    echo "Starting FastGen frontend benchmark..."
    python benchmarks/benchmark-serving/benchmark-serving.py \
        --port "$BACKEND_PORT" \
        --backend fastserve \
        --tokenizer "$TOKENIZER_PATH" \
        --dataset "$DATASET_PATH" \
        --num-prompts "$num_prompt" \
        --request-rate "$rate" \
        --process-name "$PROCESS_NAME" \
        --dataset-name "$DATASET_NAME" \
        2>&1 | tee "$LOG_FILE"

    # --- Clean up ---
    echo "Benchmark finished. Terminating backend process..."
    if ps -p $BACKEND_PID > /dev/null
    then
        kill -9 $BACKEND_PID
        echo "Backend process $BACKEND_PID terminated."
    else
        echo "Backend process $BACKEND_PID was already terminated."
    fi
    echo "======================================================================"
    echo ""
    
    # --- Wait for 30 seconds before starting the next test ---
    echo "Waiting for 30 seconds before next test..."
    sleep 30
done

echo "All tests are completed."