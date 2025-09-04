import os, sys, random, time
import dataclasses
import tqdm, subprocess
from typing import List, Tuple, Callable

import argparse
from lib.exp_result import ExpResult

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num-prompts-and-request-rates", type=str, required=True,
					 	help="List of tuples of (num prompts, request rate)s to run the experiment at, e.g. [(20, 0.1), (40, 0.2), (100, 0.8)]")
	parser.add_argument("--log-file-prefix", type=str, required=True,
					 	help="Prefix of the name of the log file")
	parser.add_argument("--log-dir", type=str, required=True,
					 	help="Directory that contains the log files")
	parser.add_argument("--latency-mult", type=float, default=1,
					 	help="Multiply latency by this factor")
	args = parser.parse_args()

	num_prompts_and_request_rates: List[Tuple[int, float]] = eval(args.num_prompts_and_request_rates)
	log_file_prefix = args.log_file_prefix
	log_dir = args.log_dir

	exp_results = []
	for num_prompts, request_rate in num_prompts_and_request_rates:
		filename = f"log/{log_dir}/{log_file_prefix}-{num_prompts}-{request_rate}-client-log"
		exp_result: ExpResult = ExpResult(num_prompts, request_rate, filename)
		exp_results.append(exp_result)

	print(exp_results)

	def print_row(selector: Callable):
		info = [selector(exp_result) for exp_result in exp_results]
		print("\t ".join([str(x) for x in info]))
	
	print_row(lambda exp_result: exp_result.throughput)
	print_row(lambda exp_result: exp_result.avg_latency*args.latency_mult)
	print_row(lambda exp_result: exp_result.avg_latency_95th*args.latency_mult)
	print_row(lambda exp_result: exp_result.per_token_latency*args.latency_mult)
	print_row(lambda exp_result: exp_result.per_output_token_latency*args.latency_mult)
	print_row(lambda exp_result: round(exp_result.per_output_token_latency_95th, 6)*args.latency_mult)

	# Collect request details
	request_details = []
	for num_prompts, request_rate in num_prompts_and_request_rates:
		filename = f"log/{log_file_prefix}-{num_prompts}-{request_rate}-client-log"
