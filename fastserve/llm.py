import time
from typing import List, Union, Optional

import asyncio
from tqdm import tqdm

from fastserve.config import ModelConfig, ParallelConfig, CacheConfig, SchedConfig
from fastserve.engine import LLMEngine
from fastserve.logger import init_logger
from fastserve.request import Request, SamplingParams


logger = init_logger(__name__)


class OfflineLLM:
    """A Large Language Model (LLM) for offline inference.
    It wraps around the LLMEngine and provides the **generate** interface to do
    offline inference on a list of prompts, which only return when all the prompts
    finish generation. If you want to do online inference where each user can asynchronously
    get the generation results in a streaming fashion, please refer to the **AsyncLLM** class.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: bool = False,
        seed: int = 1,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        max_num_blocks_per_req: int = 256,
        gpu_memory_utilization: float = 0.90,
        swap_space: int = 1,
        sched_policy: str = "fcfs",
        max_batch_size: int = 256,
        max_tokens_per_batch: int = 2048,
        profiling_file: str = None,
        use_dummy_weights: bool = False,
        proactive_offloading: bool = True,
        num_min_free_blocks_threshold: int = 0,
        num_queues_for_prediction: int = 2,
        use_skip_join: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = ModelConfig(
            model,
            tokenizer,
            trust_remote_code=trust_remote_code,
            seed=seed,
            use_dummy_weights=use_dummy_weights,
        )
        self.parallel_config = ParallelConfig(
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.cache_config = CacheConfig(
            block_size, max_num_blocks_per_req, gpu_memory_utilization, swap_space
        )
        self.sched_config = SchedConfig(
            sched_policy,
            max_batch_size,
            max_tokens_per_batch,
            model_name=model,
            profiling_file=profiling_file,
            parallel_config=self.parallel_config,
            proactive_offloading=proactive_offloading,
            num_min_free_blocks_threshold=num_min_free_blocks_threshold,
            num_queues_for_prediction=num_queues_for_prediction,
            use_skip_join=use_skip_join,
        )
        self.llm_engine = LLMEngine(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            self.sched_config,
        )

    def generate(
        self,
        prompts: Optional[Union[List[str], str]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        use_tqdm: bool = True,
    ) -> List[Request]:
        if prompts is None and prompt_token_ids is None:
            raise ValueError("prompts or prompt_token_ids must be provided")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if len(prompts) != len(prompt_token_ids):
                raise ValueError(
                    "The lengths of prompts and prompt_token_ids must be the same."
                )

        num_requests = len(prompts) if prompts is not None else len(prompt_token_ids)
        if sampling_params is None:
            sampling_params = [SamplingParams()] * num_requests
        elif isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * num_requests
        else:
            assert (
                len(sampling_params) == num_requests
            ), f"prompts should pair with the list of sampling parameters, \
                 but got {num_requests} prompts and {len(sampling_params)} sampling parameters"

        # Add requests to the engine.
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[i]
            self.llm_engine.add_request(prompt, token_ids, sampling_params[i])

        return self._run_llm_engine(use_tqdm)

    def _run_llm_engine(self, use_tqdm: bool):
        # Initialize tqdm.
        if use_tqdm:
            num_reqs = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_reqs, desc="Processed prompts")

        # Run the LLM engine until all the requests are finished.
        finished_requests = []
        while True:
            num_reqs = self.llm_engine.get_num_unfinished_requests()
            if num_reqs == 0:
                break
            _, new_finished_requests = self.llm_engine.step()
            finished_requests += new_finished_requests
            if use_tqdm:
                pbar.update(len(new_finished_requests))
        if use_tqdm:
            pbar.close()

        return finished_requests


class AsyncLLM:
    """A Large Language Model (LLM) for online inference."""

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: bool = False,
        seed: int = 1,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        max_num_blocks_per_req: int = 256,
        gpu_memory_utilization: float = 0.90,
        swap_space: int = 0,
        sched_policy: str = "fcfs",
        max_batch_size: int = 256,
        max_tokens_per_batch: int = 2048,
        profiling_file: str = None,
        use_dummy_weights: bool = False,
        proactive_offloading: bool = True,
        num_min_free_blocks_threshold: int = 0,
        num_queues_for_prediction: int = 2,
        use_skip_join: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = ModelConfig(
            model,
            tokenizer,
            trust_remote_code=trust_remote_code,
            seed=seed,
            use_dummy_weights=use_dummy_weights,
        )
        self.parallel_config = ParallelConfig(
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.cache_config = CacheConfig(
            block_size, max_num_blocks_per_req, gpu_memory_utilization, swap_space
        )
        self.sched_config = SchedConfig(
            sched_policy,
            max_batch_size,
            max_tokens_per_batch,
            model_name=model,
            profiling_file=profiling_file,
            parallel_config=self.parallel_config,
            proactive_offloading=proactive_offloading,
            num_min_free_blocks_threshold=num_min_free_blocks_threshold,
            num_queues_for_prediction=num_queues_for_prediction,
            use_skip_join=use_skip_join,
        )
        self.llm_engine = LLMEngine(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            self.sched_config,
        )
        # request_id => event
        self.request_events = {}
        # request_id => step_output
        self.step_outputs = {}
        self.is_engine_running = False
        self.kicking_request_id: Optional[str] = None
        self.timeout_interval = 1  # seconds

    async def engine_step(self, kicking_request_id: Optional[str] = None):
        """Kick the engine to process the waiting requests."""
        self.is_engine_running = True
        self.kicking_request_id = kicking_request_id
        # Yield to the event loop to allow other coroutines to run
        # while is_engine_running is True. This let the engine to add new
        # requests into the queue.
        await asyncio.sleep(0)
        step_outputs, _ = self.llm_engine.step()
        self.is_engine_running = False
        self.kicking_request_id = None

        # Notify the waiting coroutines that there are new outputs ready.
        for step_output in step_outputs:
            request_id = step_output.request_id
            # The request may get aborted
            if request_id in self.request_events:
                self.request_events[request_id].set()
                self.step_outputs[request_id] = step_output

    async def generate(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> Request:
        """Generate outputs for a single request.

        This method is a coroutine. It adds the request into the waiting queue,
        kicks the LLMEngine for generation, and streams the outputs to the caller.

        Args:
            request_id: The unique id of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            sampling_params: The sampling parameters of the request.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        if prompt is None and prompt_token_ids is None:
            raise ValueError("prompt or prompt_token_ids must be provided")

        arrival_time = time.time()
        # Create an event to notify us when there is new output from the LLMEngine.
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        # Add the request to the LLMEngine.
        self.llm_engine.add_request(
            prompt, prompt_token_ids, sampling_params, arrival_time, request_id
        )

        # Keep kicking the engine to process the requests.
        while True:
            if request_id not in self.request_events:
                # The request has been aborted.
                return

            # Kick the engine if the engine is not running.
            if not self.is_engine_running:
                try:
                    for _ in range(self.parallel_config.pipeline_parallel_size):
                        await self.engine_step(request_id)
                except RuntimeError as e:
                    await self.abort(request_id)
                    raise e

            # Wait for new output. The group_event will be set in engine_step
            # when there is new output available for the request.
            # Added a timeout to prevent deadlock.
            try:
                await asyncio.wait_for(
                    request_event.wait(), timeout=self.timeout_interval
                )
            except asyncio.TimeoutError:
                # logger.info("timeout")
                continue
            # Reset the event to wait for the next output.
            request_event.clear()

            # Decode and return new outputs.
            step_output = self.step_outputs[request_id]
            yield step_output

            # Once finished, release the resources of the request.
            if step_output.is_finished:
                logger.info(f"Finished request {request_id}.")

                del self.step_outputs[request_id]
                del self.request_events[request_id]
                # Kick the engine if the engine is not running. This is to
                # prevent that there are still requests in engine's waiting
                # queue to be executed.
                if not self.is_engine_running:
                    await self.engine_step()
                break

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return

        logger.info(f"Aborted request {request_id}.")

        self.llm_engine.abort_request(request_id)

        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.step_outputs:
            del self.step_outputs[request_id]

        # To prevent deadlock when a request is aborted while the engine is
        # running.
        if self.kicking_request_id == request_id:
            self.is_engine_running = False
            self.kicking_request_id = None
