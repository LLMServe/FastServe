"""Benchmark the latency of processing a single batch of requests.
Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_latency.py

Results on A100-40GB for OPT-125M, input_len=64, output_len=64, batch_size=64, tensor_parallel_size=1:
    FastServe latency: 0.186 seconds
    vLLM latency: 1.482 seconds
    speedup: 7.97x

Results on A100-40GB for OPT-13B, input_len=16, output_len=16, batch_size=16, tensor_parallel_size=1:
    FastServe latency: 0.411 seconds
    vLLM latency: 0.525 seconds
    speedup: 1.28x
"""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM
from vllm import SamplingParams as vllm_SamplingParams

from fastserve import OfflineLLM
from fastserve.request import SamplingParams as fs_SamplingParams


def main(args: argparse.Namespace):
    print(args)

    # Process all the requests in a single batch if possible.
    # If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    if args.system == "vllm":
        llm = LLM(
            model=args.model,
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=args.batch_size * args.input_len,
            trust_remote_code=args.trust_remote_code,
            block_size=args.block_size,
        )
    else:
        llm = OfflineLLM(
            args.model,
            tokenizer=args.tokenizer,
            max_batch_size=args.batch_size,
            max_tokens_per_batch=args.batch_size * (args.input_len + args.output_len),
            trust_remote_code=args.trust_remote_code,
            use_dummy_weights=True,
            block_size=args.block_size,
        )

    def run_to_completion(profile: bool = False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.time()

        if args.system == "vllm":
            sampling_params = vllm_SamplingParams(
                n=args.n,
                temperature=0.0 if args.use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=args.use_beam_search,
                ignore_eos=True,
                max_tokens=args.output_len,
            )
        else:
            sampling_params = fs_SamplingParams(
                n=args.n,
                temperature=0.0 if args.use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=args.use_beam_search,
                ignore_eos=True,
                max_tokens=args.output_len,
            )

        # vllm has an bug if set to [0]
        dummy_prompt_token_ids = [[10] * args.input_len] * args.batch_size
        llm.generate(
            prompt_token_ids=dummy_prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        end_time = time.time()
        latency = end_time - start_time
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("Warming up...")
    run_to_completion(profile=False)

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile=False))
    print(f"Avg latency: {np.mean(latencies)} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion."
    )
    parser.add_argument("--system", choices=["fastserve", "vllm"], default="fastserve")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters", type=int, default=3, help="Number of iterations to run."
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    args = parser.parse_args()
    main(args)
