# Throughput Benchmark

To benchmark the offline inference throughput, you should first download the ShareGPT dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```
Then run the benchmark script:
```bash
python benchmark-throughput.py --backend fastserve
python benchmark-throughput.py --backend vllm
```
The benchmark script will sample $N$ requests from the ShareGPT dataset and record the time $T$ seconds to processing them, then the throughput is calculated as $\frac{N}{T}$ r/s. Please check the parameters in `benchmark-latency.py` for the details.