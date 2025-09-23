from matplotlib import pyplot as plt
import matplotlib.axes as mpl_axes
import matplotlib
import os
import dataclasses

def xinjinization(ax: mpl_axes.Axes, remove_top_right_frame: bool=True):
	ax.tick_params('x', direction="in")
	ax.tick_params('y', direction="in")
	if remove_top_right_frame:
		ax.spines['top'].set_color(None)
		ax.spines['right'].set_color(None)

def save_fig_to_pdf(filename: str):
	if filename.endswith(".pdf"):
		filename = filename[:-4]
	plt.savefig(f"output-pdfs/{filename}_.pdf", bbox_inches='tight', transparent=True)
	os.system(f"ps2pdf -dEPSCrop output-pdfs/{filename}_.pdf output-pdfs/{filename}.pdf")
	os.remove(f"output-pdfs/{filename}_.pdf")

# SOSP's requirement
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

@dataclasses.dataclass
class ReqResult:
	prompt_len: int
	output_len: int
	ttft: float
	input_per_token_latency: float
	tpot: float
	latency: float
	per_token_latency: float

	@staticmethod
	def from_client_log(file_path: str) -> list['ReqResult']:
		with open(file_path, "r") as f:
			lines = f.readlines()
			for line in lines:
				if line.startswith("[("):
					lyst = eval(line.strip())
					result = []
					if len(lyst[0]) == 5:
						for (prompt_len, output_len, request_latency, start_time, end_time) in lyst:
							first_token_time = start_time
							ttft = first_token_time - start_time
							latency = end_time - start_time
							result.append(ReqResult(
								prompt_len,
								output_len,
								ttft,
								None,
								(end_time-start_time) / output_len,	 # latency per output token, not the same as tpot
								# (end_time-first_token_time) / (prompt_len + output_len),
								# end_time-first_token_time,
								# latency/output_len,
								latency,
								latency / (prompt_len + output_len)
							))
						return result
					else:
						for (prompt_len, output_len, request_latency, start_time, end_time, first_token_time) in lyst:
							ttft = first_token_time - start_time
							latency = end_time - start_time
							result.append(ReqResult(
								prompt_len,
								output_len,
								ttft,
								None,
								(end_time-start_time) / output_len,
								# (end_time-first_token_time) / (prompt_len + output_len),
								# end_time-first_token_time,
								# latency/output_len,
								latency,
								latency / (prompt_len + output_len)
							))
						return result
			print(f"WARN: No result found in the log file {file_path}")
			return []

def get_exp_result_root() -> str:
	assert "EXP_RESULT_ROOT" in os.environ, "Please set the environment variable EXP_RESULT_ROOT to the root of the experiment results"
	return os.environ["EXP_RESULT_ROOT"]

@dataclasses.dataclass
class ExpResult:
	num_prompts: int
	req_rate: float
	req_results: list[ReqResult]