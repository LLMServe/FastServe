import lib
from matplotlib import pyplot as plt
import dataclasses
import numpy as np
import os

os.environ["EXP_RESULT_ROOT"] = "/users/zzl/FastServe/benchmarks/artifact-evaluation/overall"

@dataclasses.dataclass
class ExpResult:
	num_prompts: int
	req_rate: float
	req_results: list[lib.ReqResult]

def read_result_series(exp_name: str, file_prefix: str) -> list[ExpResult]:
	exp_result_dir = f"{lib.get_exp_result_root()}/{exp_name}"
	file_list = os.listdir(exp_result_dir)
	file_list = [file for file in file_list if file.startswith(file_prefix) and file.endswith("-client-log")]

	results: list[ExpResult] = []
	for file in file_list:
		file_stripped = file[len(file_prefix)+len("-"):-len("-client-log")]
		num_prompts, req_rate = file_stripped.split("-")
		num_prompts = int(num_prompts)
		req_rate = float(req_rate)
		req_results = lib.ReqResult.from_client_log(exp_result_dir + "/" + file)
		if len(req_results) == 0:
			continue
		results.append(ExpResult(num_prompts, req_rate, req_results))
	
	new_results = []
	for item in results:
		invalid = False
		for item2 in results:
			if item2.req_rate <= item.req_rate and item2.num_prompts > item.num_prompts:
				invalid = True
				break
		if not invalid:
			new_results.append(item)
	new_results.sort(key=lambda x: x.req_rate)

	return new_results


@dataclasses.dataclass
class Baseline:
	label: str
	data: list[ExpResult]
	color: str
	marker: str

baselines = [
	Baseline("vLLM", read_result_series("logs", "exp-vllm-sharegpt"), "C1", "^"),
	Baseline("FastServe-FCFS", read_result_series("logs", "exp-fcfs-sharegpt"), "C2", "s"),
	Baseline("FastServe", read_result_series("logs", "exp-fastserve-sharegpt"), "C3", "d"),
]


@dataclasses.dataclass
class SLOPlotMeta:
	ttft_slo: float
	tpot_slo: float

slo_plot_metas = [
	SLOPlotMeta(1.55, 0.15),
	SLOPlotMeta(3.1, 0.3),
	SLOPlotMeta(6.2, 0.6),
]

def find_intersection_point(xs: list[float], ys: list[float], slo: float) -> float:
	for i in range(1, len(xs)):
		if ys[i] < slo:
			return xs[i-1] + (xs[i] - xs[i-1]) * (slo - ys[i-1]) / (ys[i] - ys[i-1])
	return xs[-1]

if __name__ == "__main__":
	# @dataclasses.dataclass
	# class PlotMeta:
	# 	retriver: callable
	# 	y_limit: float
	# 	x_limit: float
	# plot_metas = [
	# 	PlotMeta(lambda req_result: req_result.per_token_latency, 200, 4),
	# 	PlotMeta(lambda req_result: req_result.ttft, 40*1000, 5),
	# 	PlotMeta(lambda req_result: req_result.tpot, 50, 2.5),
	# ]

	# fig, axs = plt.subplots(1, len(plot_metas), figsize=(10, 2))
	# for fig_idx in range(len(plot_metas)):
	# 	retriver = plot_metas[fig_idx].retriver
	# 	for baseline in baselines:
	# 		x = [exp_result.req_rate for exp_result in baseline.data]
	# 		y = []
	# 		for exp_result in baseline.data:
	# 			numbers = [retriver(req_result) for req_result in exp_result.req_results]
	# 			y.append(np.percentile(numbers, 95))
	# 			# y.append(np.mean(numbers))
	# 		y = [x*1000 for x in y]
	# 		axs[fig_idx].plot(x, y, label=baseline.label, marker="o")
	# 	axs[fig_idx].set_xlabel("Request rate (req/s)")
	# 	axs[fig_idx].set_ylabel("P95 Per Token Latency (ms)")
	# 	if fig_idx == 0:
	# 		fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
	# 	if plot_metas[fig_idx].y_limit > 0:
	# 		axs[fig_idx].set_ylim(0, plot_metas[fig_idx].y_limit)
	# 	if plot_metas[fig_idx].x_limit > 0:
	# 		axs[fig_idx].set_xlim(0, plot_metas[fig_idx].x_limit)
	
	# fig.show()
	# lib.save_fig_to_pdf("p95-latency")

	for baseline in baselines:
		ttfts = [req_result.ttft for exp_result in baseline.data for req_result in exp_result.req_results]
		tpots = [req_result.tpot for exp_result in baseline.data for req_result in exp_result.req_results]
		print(baseline.label, baseline.data[0].num_prompts, baseline.data[0].req_rate, np.median(ttfts), np.median(tpots))

	target_slo = 95
	y_limit = 110
	x_limits = [4, 4, 5]
	fig, axs = plt.subplots(1, len(slo_plot_metas), figsize = (2.5*len(slo_plot_metas), 2))
	for fig_idx in range(len(slo_plot_metas)):
		ax = axs[fig_idx]
		ttft_slo = slo_plot_metas[fig_idx].ttft_slo
		tpot_slo = slo_plot_metas[fig_idx].tpot_slo
		for baseline in baselines:
			x = [exp_result.req_rate for exp_result in baseline.data]
			y = []
			for exp_result in baseline.data:
				num_ok_reqs = sum([
					1 if req_result.ttft < ttft_slo and req_result.tpot < tpot_slo else 0
					for req_result in exp_result.req_results
				])
				y.append(num_ok_reqs / len(exp_result.req_results) * 100)
			
			ax.plot(x, y, label=baseline.label, marker=baseline.marker, color=baseline.color, markersize=4)

			intersec_x = find_intersection_point(x, y, target_slo)
			ax.axvline(intersec_x, ymax=target_slo/y_limit, linestyle="--", color=baseline.color)
			ax.text(intersec_x, 50, f"{intersec_x:.2f}", ha="center", color=baseline.color)
	
		ax.axhline(target_slo, linestyle="--", color="grey")
		ax.set_xlabel("Job Arrival Rate (job/s)")
		ax.set_ylim(0, y_limit)
		ax.set_xlim(0, x_limits[fig_idx])

		if fig_idx == 0:
			ax.set_ylabel("SLO Attainment (%)")
			fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
		lib.xinjinization(ax, False)

	fig.show()
	file_path = os.path.join(lib.get_exp_result_root(), "fig13.pdf")
	plt.savefig(file_path, bbox_inches='tight', transparent=True)
