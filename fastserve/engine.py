import time
import copy
from typing import List, Optional
import asyncio

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch

from fastserve.config import ModelConfig, ParallelConfig, CacheConfig, SchedConfig
from fastserve.logger import init_logger
from fastserve.request import (
    SamplingParams,
    SamplingParams,
    Request,
    create_request,
)
from fastserve.worker import ParaWorker
from fastserve.tokenizer import get_tokenizer
from fastserve.scheduler import get_scheduler
from fastserve.utils import Counter
from fastserve.block_manager import BlockManager


logger = init_logger(__name__)


class StepOutput:
    """The output of request in one step of inference.
    It contains the information of corresponding request and the generated tokens until this step.
    """

    def __init__(self, request: Request, new_token: str, new_token_id: int):
        self.request = request
        self.request_id = request.request_id
        self.prompt = request.prompt
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = request.is_finished

    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finished})"
        )


class LLMEngine:
    """An LLMEngine launches the model executor workers and maintains runtime information.
    It receives requests from upper wrapper class and provides interface LLMEngine.step()
    to execute one iteration inference on a batch of requests chosen by the scheduler.

    Note: Users may not use LLMEngine directly, but use more user-friendly wrapper classes
    OfflineLLM and AsyncLLM instead.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: SchedConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.sched_config = sched_config

        self.request_counter = Counter()
        self.step_counter = Counter()
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )

        self.placement_groups = None
        # stages[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.stages = []

        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline = []
        self.batches_ret_futures = []

        # initialization
        self._init_placement_groups()
        self._init_workers()
        self._init_model()
        self.num_gpu_blocks, self.num_cpu_blocks = self._init_kvcache()

        self.block_manager = BlockManager(
            self.num_gpu_blocks,
            self.num_cpu_blocks,
            model_config,
            parallel_config,
            cache_config,
            self.remote_call_all_workers_async,
        )
        self.scheduler = get_scheduler(
            sched_config,
            parallel_config,
            self.block_manager,
        )
        logger.info(self.scheduler)
        logger.info(f"{self.block_manager}")

    def _init_placement_groups(self):
        if not ray.is_initialized():
            ray.init()
        num_cluster_gpus = ray.cluster_resources().get("GPU", 0)
        if (
            num_cluster_gpus
            < self.parallel_config.tensor_parallel_size
            * self.parallel_config.pipeline_parallel_size
        ):
            raise ValueError("No GPU resources available")

        self.placement_groups = []

        # for each pipeline stage, create a placement group with tensor_parallel_size GPUs
        # the 'PACK' strategy will pack the GPUs into one node
        for _ in range(self.parallel_config.pipeline_parallel_size):
            placement_group = ray.util.placement_group(
                [
                    {
                        "GPU": 1,
                    }
                ]
                * self.parallel_config.tensor_parallel_size,
                strategy="PACK",
            )
            ray.get(placement_group.ready(), timeout=1000)
            self.placement_groups.append(placement_group)
            logger.info(
                f"creating placement group with {self.parallel_config.tensor_parallel_size} GPUs"
            )

    def _init_workers(self):
        """
        for each pipeline stage, create tensor_parallel_size workers
        each worker will be assigned a GPU
        the worker will be placed in the corresponding placement group
        """
        self.pp_id = ray.put(copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id()))
        # wait until pp_id is ready
        ray.get(self.pp_id)
        init_handlers = []
        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []
            tp_id = ray.put(copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id()))
            # wait until tp_id is ready
            ray.get(tp_id)
            for j in range(self.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(self.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j
                worker = ParaWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_groups[i],
                        placement_group_bundle_index=j,
                    )
                ).remote(
                    model_config=self.model_config,
                    cache_config=self.cache_config,
                    sched_config=self.sched_config,
                    parallel_config=tmp_parallel_config,
                    pipeline_parallel_id=self.pp_id,
                    tensor_parallel_id=tp_id,
                )
                workers.append(worker)
                init_handlers.append(worker.ready.remote())
            self.stages.append(workers)
        # Ray will block until all workers are ready
        ray.get(init_handlers)

    def _init_model(self):
        """
        init model by call init_model() on all workers
        """
        self.remote_call_all_workers("init_model")

    def _init_kvcache(self):
        num_gpu_blocks, num_cpu_blocks = ray.get(
            self.stages[0][0]._profile_num_available_blocks.remote(
                self.cache_config.block_size,
                self.cache_config.gpu_memory_utilization,
                self.cache_config.cpu_swap_space,
            )
        )
        print(f"num_gpu_blocks: {num_gpu_blocks}, num_cpu_blocks: {num_cpu_blocks}")
        self.remote_call_all_workers(
            "init_kvcache_and_swap", num_gpu_blocks, num_cpu_blocks
        )
        return num_gpu_blocks, num_cpu_blocks

    def remote_call_all_workers(self, func_name: str, *args):
        """
        call func_name on all workers, blocked until all workers finish, return all the results
        """
        handlers = []
        for stage in self.stages:
            for worker in stage:
                handlers.append(getattr(worker, func_name).remote(*args))
        return ray.get(handlers)

    def remote_call_all_workers_async(self, func_name: str, *args):
        """
        call func_name asynchronously on all workers, return the futures immediately
        """
        handlers = []
        for stage in self.stages:
            for worker in stage:
                handlers.append(getattr(worker, func_name).remote(*args))
        return handlers

    def add_request(
        self,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[str]],
        sampling_params: SamplingParams,
        arrival_time: Optional[float] = None,
        request_id: Optional[int] = None,
    ):
        req = create_request(
            prompt,
            prompt_token_ids,
            sampling_params,
            self.request_counter,
            self.tokenizer,
            arrival_time,
            request_id,
        )
        self.scheduler.add_request(req)

    def step(self):
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """
        # print block usage every 100 steps
        if next(self.step_counter) % 100 == 0:
            self.block_manager.print_block_usage()

        pp_size = self.parallel_config.pipeline_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size

        # pick next batch from scheduler
        # this may trigger swap_in if some requests have been swapped out to CPU
        # this may also trigger swap_out if GPU blocks are not enough
        batched_requests = self.scheduler.get_next_batch()
        batched_requests.sort_requests_context_first()
        # logger.info(f"batched_requests: {batched_requests}")

        # allocate blocks as needed
        self.block_manager.allocate_blocks_batched(batched_requests)

        # Check if all requests are on GPU (i.e. not swapped out)
        assert self.block_manager.is_all_requests_on_gpu(
            batched_requests
        ), "Some requests are currently swapped out to CPU"

        # push the batch into pipeline
        batched_requests.start_one_iteration(time.time())
        self.batches_in_pipeline.append(batched_requests)
        remote_calls = self.remote_call_all_workers_async(
            "step",
            batched_requests.get_request_ids(),
            batched_requests.get_input_tokens_batched(),
            batched_requests.get_first_token_indexes(),
            self.block_manager.get_partial_block_table(
                batched_requests.get_request_ids()
            ),
        )
        # only the leader of the last stage return valid output, i.e., generated tokens ids
        self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        # output buffer
        step_outputs = []
        finished_reqs = []

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            generated_tokens_ids = ray.get(self.batches_ret_futures[0])
            end_time = time.time()

            # LLaMA: add a space to the front for token begins with SPIECE_UNDERLINE("▁")
            if (self.model_config.hf_config.model_type == "llama"):
                SPIECE_UNDERLINE = "▁"
                if generated_tokens_ids != [] and max(generated_tokens_ids) > self.model_config.hf_config.vocab_size:
                    print('Warning: generated token id exceeds vocab size')
                    generated_tokens_ids = [min(x, self.model_config.hf_config.vocab_size - 1) for x in generated_tokens_ids]
                _tokenlist = self.tokenizer.convert_ids_to_tokens(generated_tokens_ids)
                generated_tokens = []
                for _token in _tokenlist:
                    newstr = self.tokenizer.convert_tokens_to_string([_token,])
                    if (_token.startswith(SPIECE_UNDERLINE)):
                        newstr = " " + newstr
                    generated_tokens.append(newstr)
            else:
                generated_tokens = [
                    self.tokenizer.decode(gen_token_id, skip_special_tokens=True)
                    for gen_token_id in generated_tokens_ids
                ]

            finished_batch = self.batches_in_pipeline[0]
            finished_batch.finish_one_iteration(
                generated_tokens, generated_tokens_ids, end_time
            )
            # logger.info(f"Finished batch: {finished_batch}")

            # construct outputs
            for request, new_token, new_token_id in zip(
                finished_batch.requests, generated_tokens, generated_tokens_ids
            ):
                step_outputs.append(StepOutput(request, new_token, new_token_id))
            finished_reqs = self.scheduler.pop_finished_requests()

            # free blocks for finished requests
            self.block_manager.free_blocks_batched(finished_reqs)
            self.remote_call_all_workers_async(
                "clear_request_resource_batched", finished_reqs
            )

            # pop the finished batch
            self.batches_in_pipeline.pop(0)
            self.batches_ret_futures.pop(0)

        # proactive swapping
        self.scheduler.post_process()

        return step_outputs, finished_reqs

    def get_num_unfinished_requests(self) -> int:
        return self.scheduler.get_total_num_requests()

    def abort_request(self, request_id: int):
        self.scheduler.abort_request(request_id)
        self.block_manager.free_blocks(request_id)
        self.remote_call_all_workers_async("clear_request_resource", request_id)

    def swap_in_request(self, requests: List[Request]):
        self.block_manager.swap_in_requests(requests)

    def swap_out_request(self, requests: List[Request]):
        self.block_manager.swap_out_requests(requests)
