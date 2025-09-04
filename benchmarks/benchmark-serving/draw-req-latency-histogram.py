import matplotlib.pyplot as plt
from typing import AsyncGenerator, List, Tuple

if __name__ == "__main__":
	request_latencies: List[Tuple[int, int, float]] = eval(input())
	latencies = [latency for _, _, latency in request_latencies]

	NUM_SUBPLOTS = 4
	plot_id = 0

	# draw request latency
	plot_id += 1
	plt.subplot(1, NUM_SUBPLOTS, plot_id)
	plt.plot(latencies)
	plt.xlabel("Request index")
	plt.ylabel("Latency (s)")
	plt.title("Request latency")

	# draw histogram of request latencies
	plot_id += 1
	plt.subplot(1, NUM_SUBPLOTS, plot_id)
	plt.hist(latencies, bins=100)
	plt.xlabel("Latency (s)")
	plt.ylabel("Number of requests")
	plt.title("Histogram of request latency")

	# draw CDF of request latencies
	plot_id += 1
	plt.subplot(1, NUM_SUBPLOTS, plot_id)
	plt.hist(latencies, bins=100, cumulative=True, density=True)
	plt.xlabel("Latency (s)")
	plt.ylabel("CDF")
	plt.title("CDF of request latency")

	# draw histogram of lengths
	# plot_id += 1
	# plt.subplot(1, NUM_SUBPLOTS, plot_id)
	# lengths = [prompt_len + output_len for prompt_len, output_len, _ in request_latencies]
	# plt.hist(lengths, bins=50)
	# plt.xlabel("Length (tokens)")
	# plt.ylabel("Number of requests")
	# plt.title("Histogram of request length")

	# draw average decoding lens
	# plot_id += 1
	# plt.subplot(1, NUM_SUBPLOTS, plot_id)
	# lengths = [prompt_len + output_len/2 for prompt_len, output_len, _ in request_latencies]
	# plt.hist(lengths, bins=50)
	# plt.xlabel("Average decoding len")
	# plt.ylabel("Number of requests")
	# plt.title("Histogram of request avg decoding len")

	# draw decoding lens (use error bar graph)
	# plot_id += 1
	# plt.subplot(1, NUM_SUBPLOTS, plot_id)
	# lengths = [prompt_len + output_len/2 for prompt_len, output_len, _ in request_latencies]
	# plt.errorbar(range(len(lengths)), lengths, yerr=[output_len/2 for _, output_len, _ in request_latencies], fmt='o')
	# plt.xlabel("Request index")
	# plt.ylabel("Decoding len")
	# plt.title("Decoding len")

	# draw histogram of per-token latencies
	plot_id += 1
	plt.subplot(1, NUM_SUBPLOTS, plot_id)
	per_token_latencies = [latency / (prompt_len + output_len) for prompt_len, output_len, latency in  request_latencies]
	plt.hist(per_token_latencies, bins=100)
	plt.xlabel("Per-token latency (s)")
	plt.ylabel("Number of requests")
	plt.title("Histogram of per-token latency")

	plt.show()