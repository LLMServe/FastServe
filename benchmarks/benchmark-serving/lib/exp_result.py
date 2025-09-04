import dataclasses
import numpy as np

@dataclasses.dataclass
class RequestResult:
	prompt_len: int
	output_len: int
	latency: float
	start_time: float
	end_time: float
	
@dataclasses.dataclass
class ExpResult:
	num_prompts: int
	request_rate: float

	throughput: float
	avg_latency: float
	avg_latency_95th: float
	per_token_latency: float
	per_output_token_latency: float

	def __init__(self, num_prompts: int, request_rate: float, filepath: str):
		self.num_prompts = num_prompts
		self.request_rate = request_rate
		with open(filepath, "r") as f:
			lines = f.readlines()
			def find_line_and_parse_float(keyword):
				for line in lines:
					if keyword in line:
						for word in line.split():
							try:
								return float(word)
							except:
								pass
				raise Exception(f"Could not find {keyword} in {filepath}")
			self.throughput = find_line_and_parse_float("Throughput")
			self.avg_latency = find_line_and_parse_float("Average latency")
			self.avg_latency_95th = find_line_and_parse_float("95th percentile latency")
			self.per_token_latency = find_line_and_parse_float("Average latency per token")
			self.per_output_token_latency = find_line_and_parse_float("Average latency per output token")

			for line in lines:
				if line.startswith("[("):
					# This line contains details of every request
					request_details_list = eval(line)
					if len(request_details_list[0]) == 5:
						# The log is in the "new" format: every tuple contains (prompt_len, output_len, latency, start_time, end_time)
						self.request_details = [
							RequestResult(prompt_len, output_len, latency, start_time, end_time)
							for (prompt_len, output_len, latency, start_time, end_time) in eval(line)
						]
					else:
						# The log is in the "old" format: every tuple contains (prompt_len, output_len, latency)
						self.request_details = [
							RequestResult(prompt_len, output_len, latency, 0, 0)
							for (prompt_len, output_len, latency) in eval(line)
						]
					# Calculate 95th percentile Per Output token Latency
					self.per_output_token_latency_95th = np.percentile(
						[req.latency/req.output_len for req in self.request_details],
						95
					)
					break