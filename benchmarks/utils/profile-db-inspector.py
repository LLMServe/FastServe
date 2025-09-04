import argparse

import os, sys, matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

from fastserve.profiling import bs_config, in_len_config, bw_config
from fastserve.profiling import ProfilingDatabase, ProfilingResult

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="facebook/opt-125m")
	parser.add_argument("--db-path", type=str, required=True)
	args = parser.parse_args()

	datas = []

	profiling_database = ProfilingDatabase(args.db_path)
	profiling_result = profiling_database.get(args.model)
	for batch_size in bs_config:
		for input_len in in_len_config:
			latency_list = profiling_result.get_latency_list(1, 1, batch_size, 1, input_len)
			latency_avg = sum(latency_list) / len(latency_list)
			latency_variance = sum([(latency - latency_avg) ** 2 for latency in latency_list]) / len(latency_list)
			print(f"batch_size: {batch_size:3d}, input_len: {input_len:3d}, {latency_list}, variance: {latency_variance}")
			datas.append((batch_size, input_len, latency_avg, latency_variance))

	print("----------------")
	print("bs\tin_len\tlatency_avg\t\tlatency_variance")
	for (batch_size, input_len, latency_avg, latency_variance) in datas:
		print(f"{batch_size}\t{input_len}\t{latency_avg}\t{latency_variance}")
	
	Xs = np.unique(np.array([x[0] for x in datas])).tolist()
	Ys = np.unique(np.array([x[1] for x in datas])).tolist()
	Zs = np.zeros((len(Ys), len(Xs)))
	for (x, y, z, w) in datas:
		x = Xs.index(x)
		y = Ys.index(y)
		Zs[y][x] = z
	Xs, Ys = np.meshgrid(Xs, Ys)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(
		Xs, Ys, Zs,
		cmap=cm.coolwarm,
		linewidth=0.2,
		antialiased=True
	)
	ax.set_xlabel('batch_size')
	ax.set_ylabel('input_len')
	ax.set_zlabel('latency')
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()