import os, sys, time
import tqdm
from typing import List, Tuple

import argparse

def run_experiment(client_cmdline: str, num_prompts: int, request_rate: float):
    client_cmdline = client_cmdline.format(num_prompts=num_prompts, request_rate=request_rate)

    print(f"Using this to launch client: {client_cmdline}")

    print("Launching client...")
    client_ret_code = os.system(client_cmdline)
    if client_ret_code != 0:
        print("Client failed. Quitting...")
        sys.exit(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-cmdline", type=str, required=True,
                         help="Command line to run the client, e.g. python3 benchmark_serving.py --backend fastserve --port 9000 --dataset ~/weights/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts {num_prompts} --request-rate {request_rate} --dataset-name sharegpt --tokenizer 'facebook/opt-13b' | tee .../fastserve-{num_prompts}-{request_rate}")
    parser.add_argument("--num-prompts-req-rates", type=str, required=True,
                         help="List of tuples of (num prompts, request rate)s to run the experiment at, e.g. [(20, 0.1), (40, 0.2), (100, 0.8)]")
    args = parser.parse_args()

    client_cmdline = args.client_cmdline
    num_prompts_and_request_rates: List[Tuple[int, float]] = eval(args.num_prompts_req_rates)

    exp_results = []
    for (num_prompts, request_rate) in tqdm.tqdm(num_prompts_and_request_rates):
        print(f"\033[44mRunning with num_prompts = {num_prompts} request_rate = {request_rate}\033[0m")
        run_experiment(client_cmdline, num_prompts, request_rate)
        time.sleep(2)
    