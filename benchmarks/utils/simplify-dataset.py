"""
This script is used to simplify the dataset by only keeping X% of the samples.
I wrote this since `benchmark_serving` spends too much time on reading the whole dataset.
"""

import argparse, json
import random

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=str, required=True)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument("--ratio", type=float, required=True)
	args = parser.parse_args()

	assert args.ratio > 0 and args.ratio <= 1, "ratio must be in (0, 1]"
	random.seed(0)

	with open(args.input, "r") as f:
		data = json.load(f)
		num_total_records = len(data)
		num_records_to_keep = int(num_total_records * args.ratio)
		print(f"Keeping {num_records_to_keep} out of {num_total_records} records")
		# Select by random
		data = random.sample(data, num_records_to_keep)

	with open(args.output, "w") as f:
		json.dump(data, f)
