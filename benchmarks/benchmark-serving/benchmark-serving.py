"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (FastServe backend)
    python -m fastserve.api_server.fastserve_api_server \
        --model <your_model>

    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> \
        --dataset <target_dataset> \
        --request-rate <request_rate> \
        --process-name <process_name> \
"""
import sys
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple, Optional
import os
import pandas as pd

import aiohttp
import numpy as np
import transformers
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

# (prompt len, output len, latency, start_time, end_time)
REQUEST_LATENCY: List[Tuple[int, int, float, float, float]] = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    name: str = "sharegpt",
) -> List[Tuple[str, int, int]]:
    if name.lower() == "sharegpt":
        # Load the dataset.
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]

        # Tokenize the prompts and completions.
        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        filtered_dataset: List[Tuple[str, int, int]] = []
        for prompt, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_dataset.append((prompt, prompt_len, output_len))

        # Sample the requests.
        sampled_requests = random.sample(filtered_dataset, num_requests)
        random.shuffle(sampled_requests)
        return sampled_requests

    elif name.lower() == "alpaca":
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        # extract the input and output
        dataset = [
            (data["instruction"] + data["input"], data["output"]) for data in dataset
        ]

        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        filtered_dataset: List[Tuple[str, int, int]] = []
        for prompt, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_dataset.append((prompt, prompt_len, output_len))

        # Sample the requests.
        sampled_requests = random.sample(filtered_dataset, num_requests)
        random.shuffle(sampled_requests)
        return sampled_requests

    elif name.lower() == "mmlu":
        dataset = []
        choices = ["A", "B", "C", "D"]
        data_path = dataset_path
        subjects = sorted(
            [
                f.split("_test.csv")[0]
                for f in os.listdir(os.path.join(data_path, "test"))
                if "_test.csv" in f
            ]
        )

        for sub in subjects:
            test_df = pd.read_csv(
                os.path.join(data_path, "test", sub + "_test.csv"), header=None
            )
            for i in range(test_df.shape[0]):
                prompt = test_df.iloc[i, 0]
                k = test_df.shape[1] - 2
                for j in range(k):
                    prompt += "\n{}. {}".format(choices[j], test_df.iloc[i, j + 1])
                prompt += "\nAnswer:"
                output = test_df.iloc[i, k + 1]
                dataset.append((prompt, output))

        print("LLMU dataset size:", len(dataset))

        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        filtered_dataset: List[Tuple[str, int, int]] = []
        for prompt, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_dataset.append((prompt, prompt_len, output_len))

        # Sample the requests.
        sampled_requests = random.sample(filtered_dataset, num_requests)
        random.shuffle(sampled_requests)
        return sampled_requests

    elif name.lower() == "mix":
        sg_rate = 0.025
        dataset_path_sg = os.path.join(dataset_path, "shareGPT/sg_90k_part1.json")
        num_requests_sg = int(num_requests * sg_rate)
        dataset = sample_requests(
            dataset_path_sg, num_requests_sg, tokenizer, name="sharegpt"
        )

        dataset_path_llmu = os.path.join(dataset_path, "mmlu-data")
        num_requests_llmu = int(num_requests * (1 - sg_rate))
        dataset.extend(
            sample_requests(
                dataset_path_llmu, num_requests_llmu, tokenizer, name="mmlu"
            )
        )
        random.shuffle(dataset)

        return dataset

    elif name.lower() == "sythetic":
        prompt1 = "Hello " * 16
        prompt2 = "Hello "
        prompts = [prompt1 if _ % 2 == 0 else prompt2 for _ in range(num_requests)]
        prompt_token_ids = tokenizer(prompts).input_ids

        dataset: List[Tuple[str, int, int]] = []

        for i in range(len(prompts)):
            prompt_len = len(prompt_token_ids[i])
            output_len = 16
            dataset.append((prompts[i], prompt_len, output_len))

        return dataset

    else:
        raise ValueError(
            f"Unsupported dataset name: {name}, we currently support shareGPT and alpaca."
        )

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    process_name: str = "possion",
    request_rate: float = 1.0,
    cv: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    interval_lens = len(input_requests)
    input_requests = iter(input_requests)

    if request_rate not in [float("inf"), 0.0]:
        if process_name == "uniform":
            intervals = [1.0 / request_rate for _ in range(interval_lens)]
        elif process_name == "gamma":
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        elif process_name == "possion":
            cv = 1
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        else:
            raise ValueError(
                f"Unsupported prosess name: {process_name}, we currently support uniform, gamma and possion."
            )
    for idx, request in enumerate(input_requests):
        yield request
        if request_rate == float("inf") or request_rate == 0.0:
            continue

        interval = intervals[idx]
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

sent_pbar: Optional[tqdm] = None
finish_pbar: Optional[tqdm] = None
last_print_time = 0
last_print_num_outputs = 0


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    request_cv: float = 1.0,
    process_name: str = "possion",
) -> None:
    async def send_request(
        backend: str,
        api_url: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        best_of: int,
        use_beam_search: bool,
    ) -> None:
        request_start_time = time.time()

        headers = {"User-Agent": "Benchmark Client"}
        if backend == "fastserve" or backend == "vllm" or backend == "fastertransformer":
            pload = {
                "prompt": prompt,
                "n": 1,
                "best_of": best_of,
                "use_beam_search": use_beam_search,
                "temperature": 0.0 if use_beam_search else 1.0,
                "top_p": 1.0,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": False,
            }
        elif backend == "tgi":
            assert not use_beam_search
            params = {
                "best_of": best_of,
                "max_new_tokens": output_len,
                "do_sample": True,
            }
            pload = {
                "inputs": prompt,
                "parameters": params,
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # print(f"Sending the request: {pload}")
        global sent_pbar, finish_pbar, last_print_time, last_print_num_outputs
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            sent_pbar.update(1)
            sent_pbar.refresh()
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # if output['text'][:len(prompt)] != prompt:
            #     print('(Prompt mismatch)', prompt.__repr__(), output['text'].__repr__())
            # else:
            #     print(prompt.__repr__(), output['text'][len(prompt):].__repr__())

            # Re-send the request if it failed.
            if "error" in output:
                print(f"Failed to process the request: {output['error']}, request: {pload}")
                assert False

            request_end_time = time.time()
            request_latency = request_end_time - request_start_time
            REQUEST_LATENCY.append((prompt_len, output_len, request_latency, request_start_time, request_end_time))

            finish_pbar.update(1)
            finish_pbar.refresh()

            if len(REQUEST_LATENCY)-last_print_num_outputs > len(input_requests)*0.1 or \
                time.time() - last_print_time > 30:
                if last_print_time != 0:
                    print("\n\n")
                    print(f"{sent_pbar.n} requests sent, {len(REQUEST_LATENCY)} requests finished.")
                    print(f"Gap: {sent_pbar.n-finish_pbar.n} / {sent_pbar.n} ({(sent_pbar.n-finish_pbar.n)/sent_pbar.n*100:.2f}%)")
                    print("")
                    sys.stdout.flush()
                
                sent_pbar.refresh()
                finish_pbar.refresh()
                last_print_time = time.time()
                last_print_num_outputs = len(REQUEST_LATENCY)

    tasks: List[asyncio.Task] = []
    async for request in get_request(
        input_requests, process_name, request_rate, request_cv
    ):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                prompt,
                prompt_len,
                output_len,
                best_of,
                use_beam_search,
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(
        args.dataset, args.num_prompts, tokenizer, args.dataset_name
    )
    print("Sampling done. Start benchmarking...")

    global sent_pbar, finish_pbar
    sent_pbar = tqdm(total=args.num_prompts, desc="Sent", colour="#ee0000")
    finish_pbar = tqdm(total=args.num_prompts, desc="Done", colour="#66ccff")
    benchmark_start_time = time.time()
    asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
            args.request_cv,
            args.process_name,
        )
    )
    benchmark_end_time = time.time()
    sent_pbar.close()
    finish_pbar.close()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")
    print(REQUEST_LATENCY)
    # for i in REQUEST_LATENCY:
    #     print(i)
    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency, _, _ in REQUEST_LATENCY])
    p95_latency = np.percentile([latency for _, _, latency, _, _ in REQUEST_LATENCY], 95)
    print(f"Average latency: {avg_latency:.2f} s")
    print(f"95th percentile latency: {p95_latency:.2f} s")

    # Current latency interface doese not support p95 token level latency
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency, _, _ in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency:.5f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency, _, _ in REQUEST_LATENCY]
    )
    print("Average latency per output token: " f"{avg_per_output_token_latency:.3f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend", type=str, default="fastserve", choices=["fastserve", "vllm", "tgi", "fastertransformer"]
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--request-cv",
        type=float,
        default=1.0,
        help="the coefficient of variation of the gap between" "the requests.",
    )
    parser.add_argument(
        "--process-name",
        type=str,
        default="possion",
        choices=["possion", "gamma", "uniform"],
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "alpaca", "sythetic", "mmlu", "mix"],
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    args = parser.parse_args()
    main(args)
