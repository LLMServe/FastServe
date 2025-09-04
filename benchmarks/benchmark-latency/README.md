# Latency Benchmark

To benchmark the latency of processing a single batch of requests (each request has the same input length and output length), run:
```bash
python benchmark-latency.py --system fastserve
python benchmark-latency.py --system vllm
```
You can change the input-len, output-len, batch-size for benchmarking. Please check the parameters in `benchmark-latency.py`.

For FasterTransformer, please refer to `fastertransformer-latency/README.md`.
