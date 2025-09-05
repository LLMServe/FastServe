# Artifact Evaluation
This is the artifact evaluation of NSDI'26 paper "Iteration-Level Preemptive Scheduling for Large Language Model Inference". The experiments in this artifact are designed for a single-GPU setup.

## Env
After you `ssh` into machine (e.g., xinjin1), run the following command to activate the conda environment.
```
source ~/.bashrc
```

### Figure 11 & Figure 12
```
# FastGen backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 10000 --model /users/wby/weights/opt-13b-swtransformer --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy sj-mlfq --max-batch-size 16 --max-tokens-per-batch 2048 --use-dummy-weights --proactive-offloading --use-skip-join --profiling-file ./profiling-model

# FastGen frontend
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer /users/wby/weights/opt-13b-swtransformer \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 400 \
        --request-rate 2.5 \
        --process-name possion \
        --dataset-name sharegpt
```

```
# FastGen-FCFS backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 10000 --model /users/wby/weights/opt-13b-swtransformer --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy fcfs --max-batch-size 16 --max-tokens-per-batch 2048 --use-dummy-weights --proactive-offloading --use-skip-join --profiling-file ./profiling-model

# FastGen-FCFS frontend
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer /users/wby/weights/opt-13b-swtransformer \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 400 \
        --request-rate 2.5 \
        --process-name possion \
        --dataset-name sharegpt
```

```
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
       --max-num-seqs 16 \
       --enforce-eager \
       --disable-custom-all-reduce \
       --device cuda \
       --num-scheduler-steps 1 \
       --swap-space 4 \
       --disable-log-requests \
       --num-gpu-blocks-override 400 \

# vllm frontend
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 7845 \
        --backend vllm \
        --tokenizer /users/wby/weights/opt-13b \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 400 \
        --request-rate 2.5 \
        --process-name possion \
        --dataset-name sharegpt
```

```
# vllm backend with chunked-prefill
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
       --max-num-seqs 16 \
       --enforce-eager \
       --disable-custom-all-reduce \
       --device cuda \
       --num-scheduler-steps 1 \
       --swap-space 4 \
       --disable-log-requests \
       --num-gpu-blocks-override 400 \
       --enable-chunked-prefill

# vllm frontend with chunked-prefill
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 7845 \
        --backend vllm \
        --tokenizer /users/wby/weights/opt-13b \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 400 \
        --request-rate 2.5 \
        --process-name possion \
        --dataset-name sharegpt
```

### Figure 14
```
# FastGen backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 10000 --model /users/wby/weights/opt-13b-swtransformer --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy sj-mlfq --max-batch-size 128 --max-tokens-per-batch 2048 --use-dummy-weights --proactive-offloading --use-skip-join --profiling-file ./profiling-model

# FastGen frontend
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer /users/wby/weights/opt-13b-swtransformer \
        --dataset ~/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000 \
        --request-rate 1 \
        --process-name possion \
        --dataset-name sharegpt


# FastGen-FCFS backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 10000 --model /users/wby/weights/opt-13b-swtransformer --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy fcfs --max-batch-size 64 --max-tokens-per-batch 2048 --use-dummy-weights --proactive-offloading --use-skip-join --profiling-file ./profiling-model

# FastGen-FCFS frontend
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer /users/wby/weights/opt-13b-swtransformer \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000 \
        --request-rate 1 \
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
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 7845 \
        --backend vllm \
        --tokenizer /users/wby/weights/opt-13b \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000 \
        --request-rate 1 \
        --process-name possion \
        --dataset-name sharegpt
```

### Figure 15
```
# FastServe-FCFS backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 8000 --model ~/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/ --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy fcfs --max-batch-size 16 --max-tokens-per-batch 2048 --use-dummy-weights

# FastGen-FCFS frontend
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer ~/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/ \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000 \
        --request-rate 1 \
        --process-name possion \
        --dataset-name sharegpt

# vLLM backend
conda activate fastserve-rr-vllm
python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 9000 --model ~/research/weights/Llama3-8b-CunnyGPT-16bit/ --load-format dummy --dtype float16 --block-size 16 --gpu-memory-utilization 0.9 --max-num-batched-tokens 2048 --max-model-len 2048 --max-num-seqs 16 --enforce-eager --disable-custom-all-reduce --device cuda --num-scheduler-steps 1 --swap-space 32 --disable-log-requests

# vllm frontend
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 7845 \
        --backend vllm \
        --tokenizer ~/research/weights/Llama3-8b-CunnyGPT-16bit/ \
        --dataset /users/zzl/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000 \
        --request-rate 1 \
        --process-name possion \
        --dataset-name sharegpt

# FastGen backend
conda activate fastserve $$ cd FastServe
python3 fastserve/api_server/fastserve_api_server.py --host 0.0.0.0 --port 10000 --model ~/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/ --block-size 16 --gpu-memory-utilization 0.9 --swap-space 32 --sched-policy sj-mlfq --max-batch-size 16 --max-tokens-per-batch 2048 --use-dummy-weights --proactive-offloading --use-skip-join --profiling-file ./profiling-model

# FastGen frontend
conda activate fastserve $$ cd FastServe
python benchmarks/benchmark-serving/benchmark-serving.py \
        --port 10000 \
        --backend fastserve \
        --tokenizer ~/research/weights/Llama3-8b-CunnyGPT-16bit-swifttransformer/ \
        --dataset ~/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000 \
        --request-rate 1 \
        --process-name possion \
        --dataset-name sharegpt
```