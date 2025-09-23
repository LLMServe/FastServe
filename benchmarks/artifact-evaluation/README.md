# Artifact Evaluation
This is the artifact evaluation of NSDI'26 paper "Iteration-Level Preemptive Scheduling for Large Language Model Inference". The experiments in this artifact are designed for a single-GPU setup.

## Env
After you `ssh` into machine (e.g., xinjin1), run the following command to activate the conda environment.
```
source ~/.bashrc
```

### Figure 11 & Figure 12 & Figure 13 & Figure 18
First, navigate into the `overall` directory which contains the necessary shell scripts. Next, execute all the `overall_*.sh` scripts to run the experiments. These scripts will generate the log data required for the figures. (Note: You can modify the output directory variable inside each `.sh` script to specify where you want to save the log files.) Once all the benchmark scripts have successfully completed, run the `fig*.py` Python scripts to process the logs and generate the final figures. (Note: Ensure the log directory path in the Python scripts matches the location where the benchmark logs were saved.)

```
cd overall
bash overall_fastserve_sharegpt.sh
bash overall_vllm_sharegpt.sh
bash overall_fcfs_sharegpt.sh
bash overall_cp_sharegpt.sh
bash overall_fastserve_alpaca.sh
bash overall_vllm_alpaca.sh
bash overall_fcfs_alpaca.sh
bash overall_cp_alpaca.sh

# Fig 11
python fig11_sharegpt.py
python fig11_alpaca.py

# Fig 12
python fig12.py

# Fig 13
python fig13.py

# Fig 18
python fig18_sharegpt.py
python fig18_alpaca.py

```

### Figure 14
```
# FastGen backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 10000 --model /users/wby/weights/opt-13b-swtransformer --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy sj-mlfq --max-batch-size 128 --max-tokens-per-batch 2048 --use-dummy-weights --proactive-offloading --use-skip-join --profiling-file ./profiling-model

# FastGen frontend
conda activate fastserve $$ cd FastServe
# Define the lists of parameters as shell arrays
num_prompt_list=(200 400 500 550 600 650 700 750)
rate_list=(1 2 2.5 2.75 3 3.25 3.5 3.75)
# Get the total number of elements in the array
num_tests=${#rate_list[@]}
# Loop from 0 to the number of tests - 1
for (( i=0; i<num_tests; i++ )); do
    # Get the parameters for the current run
    num_prompt=${num_prompt_list[i]}
    rate=${rate_list[i]}

    # Print a message indicating which test is running
    echo "======================================================================"
    echo "Running Test $((i+1))/$num_tests: Rate = $rate, Num Prompts = $num_prompt"
    echo "======================================================================"

    # Execute the benchmark command with the current parameters
    python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer /users/wby/weights/opt-13b-swtransformer \
        --dataset ~/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts "$num_prompt" \
        --request-rate "$rate" \
        --process-name possion \
        --dataset-name sharegpt


# FastGen-FCFS backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 10000 --model /users/wby/weights/opt-13b-swtransformer --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy fcfs --max-batch-size 64 --max-tokens-per-batch 2048 --use-dummy-weights --proactive-offloading --use-skip-join --profiling-file ./profiling-model

# FastGen-FCFS frontend
conda activate fastserve $$ cd FastServe
# Define the lists of parameters as shell arrays
num_prompt_list=(200 400 500 550 600 650 700 750)
rate_list=(1 2 2.5 2.75 3 3.25 3.5 3.75)
# Get the total number of elements in the array
num_tests=${#rate_list[@]}
# Loop from 0 to the number of tests - 1
for (( i=0; i<num_tests; i++ )); do
    # Get the parameters for the current run
    num_prompt=${num_prompt_list[i]}
    rate=${rate_list[i]}

    # Print a message indicating which test is running
    echo "======================================================================"
    echo "Running Test $((i+1))/$num_tests: Rate = $rate, Num Prompts = $num_prompt"
    echo "======================================================================"

    # Execute the benchmark command with the current parameters
    python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer /users/wby/weights/opt-13b-swtransformer \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts "$num_prompt" \
        --request-rate "$rate" \
        --process-name possion \
        --dataset-name sharegpt

# vllm backend
conda activate fastserve-rr-vllm
python -m vllm.entrypoints.api_server \
       --host 0.0.0.0 --port 7845 \
       --model /users/zzl/models/opt-13b \
       --tokenizer /users/zzl/models/opt-13b \
       --load-format dummy \
       --dtype float16 \
       --block-size 16 \
       --gpu-memory-utilization 0.9 \
       --max-num-batched-tokens 2048 \
       --max-model-len 2048 \
       --max-num-seqs 64 \
       --enforce-eager \
       --disable-custom-all-reduce \
       --device cuda \
       --num-scheduler-steps 1 \
       --swap-space 4 \
       --disable-log-requests \
       --num-gpu-blocks-override 400 \

# vllm frontend
conda activate fastserve-rr-vllm
# Define the lists of parameters as shell arrays
num_prompt_list=(200 400 500 550 600 650 700 750)
rate_list=(1 2 2.5 2.75 3 3.25 3.5 3.75)
# Get the total number of elements in the array
num_tests=${#rate_list[@]}
# Loop from 0 to the number of tests - 1
for (( i=0; i<num_tests; i++ )); do
    # Get the parameters for the current run
    num_prompt=${num_prompt_list[i]}
    rate=${rate_list[i]}

    # Print a message indicating which test is running
    echo "======================================================================"
    echo "Running Test $((i+1))/$num_tests: Rate = $rate, Num Prompts = $num_prompt"
    echo "======================================================================"

    # Execute the benchmark command with the current parameters
    python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 7845 \
        --backend vllm \
        --tokenizer /users/wby/weights/opt-13b \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts "$num_prompt" \
        --request-rate "$rate" \
        --process-name possion \
        --dataset-name sharegpt
```

### Figure 15
```
# FastServe-FCFS backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 8000 --model /user/lsy/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/ --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy fcfs --max-batch-size 16 --max-tokens-per-batch 2048 --use-dummy-weights

# FastGen-FCFS frontend
conda activate fastserve $$ cd FastServe
# Define the lists of parameters as shell arrays
num_prompt_list=(200 400 600 660 700 760 800 6000 1000 8000 1200 1400 1600 1800 2000 2200 3600)
rate_list=(1 2 3 3.3 3.5 3.8 4 4 5 5.5 6 7 8 9 10 11 12)
# Get the total number of elements in the array
num_tests=${#rate_list[@]}
# Loop from 0 to the number of tests - 1
for (( i=0; i<num_tests; i++ )); do
    # Get the parameters for the current run
    num_prompt=${num_prompt_list[i]}
    rate=${rate_list[i]}

    # Print a message indicating which test is running
    echo "======================================================================"
    echo "Running Test $((i+1))/$num_tests: Rate = $rate, Num Prompts = $num_prompt"
    echo "======================================================================"

    # Execute the benchmark command with the current parameters
    python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer /user/lsy/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/ \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts "$num_prompt" \
        --request-rate "$rate" \
        --process-name possion \
        --dataset-name sharegpt

# vLLM backend
conda activate fastserve-rr-vllm
python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 9000 --model /user/lsy/research/weights/Llama3-8b-CunnyGPT-16bit/ --load-format dummy --dtype float16 --block-size 16 --gpu-memory-utilization 0.9 --max-num-batched-tokens 2048 --max-model-len 2048 --max-num-seqs 16 --enforce-eager --disable-custom-all-reduce --device cuda --num-scheduler-steps 1 --swap-space 32 --disable-log-requests

# vllm frontend
conda activate fastserve-rr-vllm
# Define the lists of parameters as shell arrays
num_prompt_list=(200 400 600 660 700 760 800 6000 1000 8000 1200 1400 1600 1800 2000 2200 3600)
rate_list=(1 2 3 3.3 3.5 3.8 4 4 5 5.5 6 7 8 9 10 11 12)
# Get the total number of elements in the array
num_tests=${#rate_list[@]}
# Loop from 0 to the number of tests - 1
for (( i=0; i<num_tests; i++ )); do
    # Get the parameters for the current run
    num_prompt=${num_prompt_list[i]}
    rate=${rate_list[i]}

    # Print a message indicating which test is running
    echo "======================================================================"
    echo "Running Test $((i+1))/$num_tests: Rate = $rate, Num Prompts = $num_prompt"
    echo "======================================================================"

    # Execute the benchmark command with the current parameters
    python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 7845 \
        --backend vllm \
        --tokenizer /user/lsy/research/weights/Llama3-8b-CunnyGPT-16bit/ \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts "$num_prompt" \
        --request-rate "$rate" \
        --process-name possion \
        --dataset-name sharegpt

# FastGen backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 10000 --model /user/lsy/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/ --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy sj-mlfq --max-batch-size 16 --max-tokens-per-batch 2048 --use-dummy-weights --proactive-offloading --use-skip-join --profiling-file ./profiling-model

# FastGen frontend
conda activate fastserve $$ cd FastServe
# Define the lists of parameters as shell arrays
num_prompt_list=(200 400 600 660 700 760 800 6000 1000 8000 1200 1400 1600 1800 2000 2200 3600)
rate_list=(1 2 3 3.3 3.5 3.8 4 4 5 5.5 6 7 8 9 10 11 12)
# Get the total number of elements in the array
num_tests=${#rate_list[@]}
# Loop from 0 to the number of tests - 1
for (( i=0; i<num_tests; i++ )); do
    # Get the parameters for the current run
    num_prompt=${num_prompt_list[i]}
    rate=${rate_list[i]}

    # Print a message indicating which test is running
    echo "======================================================================"
    echo "Running Test $((i+1))/$num_tests: Rate = $rate, Num Prompts = $num_prompt"
    echo "======================================================================"

    # Execute the benchmark command with the current parameters
    python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer /user/lsy/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/ \
        --dataset ~/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts "$num_prompt" \
        --request-rate "$rate" \
        --process-name possion \
        --dataset-name sharegpt
```

## Visualization
The plotting code is in [plot.ipynb](./plot.ipynb). You can run it in a Jupyter notebook environment. If you want to use your own evaluation results, please modify it according to the comments in the notebook.