import argparse
import os, sys, dataclasses, time
from uuid import uuid4
import threading
from typing import List, Optional, Tuple, Union, Dict
from queue import Queue
import asyncio
from mpi4py import MPI

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

dir_path = os.path.dirname(os.path.realpath(__file__))
FASTER_TRANSFORMER_PATH = os.path.join(dir_path, "../../reference_proj/fastertransformer_ref")
sys.path.append(FASTER_TRANSFORMER_PATH)

import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
from examples.pytorch.gpt.utils import comm
from examples.pytorch.gpt.utils import gpt_decoder
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

@dataclasses.dataclass
class ModelHyperparam:
	vocab_size: int
	max_position_embeddings: int
	hidden_size: int
	num_layers: int
	num_heads: int
	head_dim: int
	ffn_inter_dim: int

MODEL_HYPERPARAM_MAP = {
	"opt-1.3b": ModelHyperparam(
		vocab_size = 50272,
		max_position_embeddings = 2048,
		hidden_size = 2048,
		num_layers = 24,
		num_heads = 32,
		head_dim = 64,
		ffn_inter_dim = 8192
	),
	"opt-13b": ModelHyperparam(
		vocab_size = 50272,
		max_position_embeddings = 2048,
		hidden_size = 5120,
		num_layers = 40,
		num_heads = 40,
		head_dim = 128,
		ffn_inter_dim = 20480
	),
	"opt-30b": ModelHyperparam(
		vocab_size = 50272,
		max_position_embeddings = 2048,
		hidden_size = 7168,
		num_layers = 48,
		num_heads = 56,
		head_dim = 128,
		ffn_inter_dim = 28672
	),
	"opt-66b": ModelHyperparam(
		vocab_size = 50272,
		max_position_embeddings = 2048,
		hidden_size = 9216,
		num_layers = 64,
		num_heads = 72,
		head_dim = 128,
		ffn_inter_dim = 36864
	),
	"opt-175b": ModelHyperparam(
		vocab_size = 50272,
		max_position_embeddings = 2048,
		hidden_size = 12288,
		num_layers = 96,
		num_heads = 96,
		head_dim = 128,
		ffn_inter_dim = 49152
	),
	"opt-175b-half": ModelHyperparam(	# Only have half of the layers compared to opt-175b
		vocab_size = 50272,
		max_position_embeddings = 2048,
		hidden_size = 12288,
		num_layers = 48,
		num_heads = 96,
		head_dim = 128,
		ffn_inter_dim = 49152
	)
}

@torch.no_grad()
def run_inference_on_batch(gpt: ParallelGPT, output_len: int, input_token_ids: List[List[int]]) -> List[List[int]]:
	"""Run inference on a batch of inputs"""

	batch_size = len(input_token_ids)
	infer_decode_args = dict(
        beam_width = 1,
        top_k = 1 * torch.ones(batch_size, dtype=torch.int32),
        top_p = 0.0 * torch.ones(batch_size, dtype=torch.float32),
        temperature = 1.0 * torch.ones(batch_size, dtype=torch.float32),
        repetition_penalty = None,
        presence_penalty = None,
        beam_search_diversity_rate = 0.0 * torch.ones(batch_size, dtype=torch.float32),
        len_penalty = 0.0 * torch.ones(size=[batch_size], dtype=torch.float32),
        bad_words_list = None,
        min_length = 0 * torch.ones(size=[batch_size], dtype=torch.int32),
        random_seed = torch.zeros([batch_size], dtype=torch.int64)
    )

	start_lengths = torch.IntTensor([len(ids) for ids in input_token_ids])

	input_token_ids = [torch.IntTensor(ids) for ids in input_token_ids]
	input_token_ids = pad_sequence(input_token_ids, batch_first=True, padding_value=2)

	print(f"({MPI.COMM_WORLD.Get_rank()}) Forwarding. batch_size = {batch_size}, input lengths = {start_lengths.tolist()}, output_len = {output_len}")

	# max_start_length = max(start_lengths)
	# start_lengths = torch.IntTensor([max_start_length] * batch_size)

	outputs = gpt(
		input_token_ids,
		start_lengths,
		output_len,
		return_output_length = True,
		return_cum_log_probs = False,
		**infer_decode_args
		# TODO Ignore EOS here
	)

	return outputs[0].cpu().tolist()


app = FastAPI()

@dataclasses.dataclass
class WaitingRequest:
	uuid: str
	input_str: str
	output_len: int
	event: threading.Event

waiting_requests = Queue()
max_batch_size: int = 0
gpt: Optional[ParallelGPT] = None
enc: Optional[encoder.Encoder] = None
request_results = {}	# uuid -> output_token_ids

@app.post("/generate")
async def generate(request: Request) -> Response:
	"""Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not. (should always be False)
    - other fields: the sampling parameters
		- max_tokens
    """
	request_uuid = str(uuid4().hex)
	print("Received a request", request_uuid)
	request_dict = await request.json()
	prompt = request_dict.pop("prompt")
	stream = request_dict.pop("stream", False)
	max_tokens = request_dict.pop("max_tokens")
	assert not stream, "stream should always be False"

	event = threading.Event()
	waiting_requests.put(WaitingRequest(
		uuid = request_uuid,
		input_str = prompt,
		output_len = max_tokens,
		event = event
	))

	# Wait for the event to be set
	while not event.is_set():
		# "Busy wait"
		# We cannot use event.wait() because it will block the main thread entirely
		await asyncio.sleep(0.1)
	
	# Get the result
	output_token_ids = request_results.pop(request_uuid)
	print("Finished request", request_uuid)

	result = {"text": str(output_token_ids)}
	return JSONResponse(result)

def on_tick(max_tokens_per_batch: int):
	comm = MPI.COMM_WORLD
	if comm.Get_rank() == 0:
		"""The main thread scans the waiting_requests list over and over, and runs
		forward when it is not empty"""
		cur_requests = []
		max_prompt_len = 0
		max_output_len = 0
		while len(cur_requests) < max_batch_size and waiting_requests.empty() == False:
			with waiting_requests.mutex:
				new_request: Request = waiting_requests.queue[0]
			new_request_prompt_len = len(enc.encode(new_request.input_str))
			new_max_prompt_len = max(new_request_prompt_len, max_prompt_len)
			new_max_output_len = max(new_request.output_len, max_output_len)
			if (new_max_prompt_len+new_max_output_len)*(len(cur_requests)+1) > max_tokens_per_batch:
				break

			cur_requests.append(new_request)
			max_prompt_len = new_max_prompt_len
			max_output_len = new_max_output_len
			waiting_requests.get()

		if cur_requests == []:
			return
		input_token_ids = [enc.encode(request.input_str) for request in cur_requests]
		output_len = max([req.output_len for req in cur_requests])

		# Shrink output_len if max(input_token_ids' length) + output_len >= 2048
		max_input_len = max([len(ids) for ids in input_token_ids])
		if max_input_len + output_len >= 2048:
			output_len = 2048 - max_input_len
			print(f"Shrinking output_len to {output_len}")

		print(f"Picked {len(cur_requests)} requests. {waiting_requests.qsize()} requests still waiting")

		comm = MPI.COMM_WORLD
		comm.bcast(input_token_ids, root=0)
		comm.bcast(output_len, root=0)

		output_token_ids = run_inference_on_batch(gpt, output_len, input_token_ids)

		for request in cur_requests:
			request_results[request.uuid] = output_token_ids.pop(0)
			request.event.set()
	else:
		input_token_ids = comm.bcast(None, root=0)
		output_len = comm.bcast(None, root=0)
		output_token_ids = run_inference_on_batch(gpt, output_len, input_token_ids)
				

def main_loop(max_tokens_per_batch: int):
	while True:
		# print(f"Tick {MPI.COMM_WORLD.Get_rank()}")
		on_tick(max_tokens_per_batch)
		time.sleep(0.1)

@torch.no_grad()
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--host", type=str, default="localhost")
	parser.add_argument("--port", type=int, required=True,
					 help="The port number to listen to")
	parser.add_argument("--model-name", type=str, required=True,
					 help="The model name, e.g. opt-13b")
	parser.add_argument("--tensor-para-size", type=int, required=True,
					 help="The tensor parallel size, e.g. 1, 2, 4, 8")
	parser.add_argument("--pipeline-para-size", type=int, required=True,
					 help="The pipeline parallel size, e.g. 1, 2, 4")
	parser.add_argument("--max-batch-size", type=int, required=True,
					 help="The maximum batch size")
	parser.add_argument('--vocab-file', type=str, required=True,
					 help='path to the vocabulary file (gpt2-vocab.json)')
	parser.add_argument('--merge-file', type=str, required=True,
					 help='path to the merge file (gpt2-merges.txt)')
	parser.add_argument('--lib-path', type=str, default='./lib/libth_transformer.so',
                     help='path to the pyt_fastertransformer dynamic lib file')
	parser.add_argument('--inference-dtype', type=str, default='fp16',
					 help='inference dtype, fp16 or fp32')
	parser.add_argument("--max-tokens-per-batch", type=int, required=True,
					 help="The maximum number of tokens per batch. Set carefully to avoid CUDA OOM")
	args = parser.parse_args()
	print(args)

	# Load model hyperparameters
	assert args.model_name in MODEL_HYPERPARAM_MAP, f"model_name {args.model_name} not in MODEL_HYPERPARAM_MAP"
	model_hyperparam = MODEL_HYPERPARAM_MAP[args.model_name]

	# Save max_batch_size to global var
	global max_batch_size
	max_batch_size = args.max_batch_size

	# Initialize communicator
	assert args.pipeline_para_size == 1, "pipeline parallel is not supported yet"
	assert args.tensor_para_size == MPI.COMM_WORLD.Get_size(), "tensor parallel size must be equal to world size"

	# Check FMHA_ENABLE
	if "FMHA_ENABLE" not in os.environ or os.environ["FMHA_ENABLE"] != "ON":
		print("Pay attention: FMHA_ENABLE environment variable is not set to ON. FastTransformer will use unfused multihead attention")
		print("If you want to use fused multihead attention, please set FMHA_ENABLE=ON in your environment, e.g. by running export FMHA_ENABLE=ON in your shell")
		assert False
	else:
		print("FMHA_ENABLE environment variable is set to ON. FastTransformer will use fused multihead attention")

	assert MPI.Is_initialized(), "MPI is not initialized, please use mpirun to run this script"
	# Load encoder
	global enc
	enc = encoder.get_encoder(args.vocab_file, args.merge_file)

	# Load GPT
	global gpt
	gpt = ParallelGPT(
		model_hyperparam.num_heads,
		model_hyperparam.head_dim,
		model_hyperparam.vocab_size,
		0, 0,
		model_hyperparam.num_layers,
		model_hyperparam.max_position_embeddings,
		args.tensor_para_size,
		args.pipeline_para_size,
		lib_path = args.lib_path,
		inference_data_type = "fp16",
		int8_mode = False,
		weights_data_type = "fp16",
		shared_contexts_ratio = 0.0,
		gpt_with_moe = False,
		expert_num = 0,
		moe_k = 0,
		moe_layer_index = []
	)

	# gpt.load(
	# 	checkpoint_path = "Dummy",
	# 	inference_data_type = args.inference_dtype,
	# 	config = None,
	# 	device = device,
	# 	use_dummy_weight = True
	# )

	# Warm-up
	result = run_inference_on_batch(gpt, 10, [[1,2,3,4,5,6,7,8,9,10,11,12], [5,6], [8,9,10], [2333]])
	print(result)
	torch.cuda.empty_cache()

	if MPI.COMM_WORLD.Get_rank() == 0:
		threading.Thread(target=lambda: main_loop(args.max_tokens_per_batch), daemon=True).start()
		print("Starting the server")
		uvicorn.run(
			app,
			host=args.host,
			port=args.port,
			log_level="debug",
			timeout_keep_alive=5
		)
	else:
		main_loop(args.max_tokens_per_batch)


if __name__ == "__main__":
	main()
