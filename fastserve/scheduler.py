from abc import ABC, abstractmethod
import copy
from typing import List, Callable, Tuple
import warnings
import torch
import time

from fastserve.config import SchedConfig, ParallelConfig
from fastserve.logger import init_logger
from fastserve.request import Request, BatchedRequests
from fastserve.profiling import ProfilingDatabase
from fastserve.block_manager import BlockManager, BlockLocation

logger = init_logger(__name__)


class Scheduler(ABC):
    """The abstract class for a scheduler.
    It should maintain all the requests in the current systems and their
    runtime statistics which are needed for scheduling. Before each iteration
    begins, the LLMEngine will call get_next_batch() method to get a
    BatchedRequets object for the next iteration. After each iteration ends,
    the LLMEngine will call the pop_finished_requests() method to get the
    finished requests in the current iteration.
    """

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """
        Add a request to the scheduler.
        """
        raise NotImplementedError()

    @abstractmethod
    def abort_request(self, request_id: int) -> None:
        """
        Abort a request from the scheduler.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_next_batch(self) -> BatchedRequests:
        """
        Get a batch of requests for the execution of next iteration.
        """
        raise NotImplementedError()

    @abstractmethod
    def pop_finished_requests(self) -> List[Request]:
        """
        Pop the finished requests from the scheduler.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_total_num_requests(self) -> int:
        """
        Get the total number of requests in the system.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_processing_num_requests(self) -> int:
        """
        Get the number of requests that are being processed.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_waiting_num_requests(self) -> int:
        """
        Get the number of requests that are waiting for processing.
        """
        raise NotImplementedError()

    def post_process(self) -> None:
        """
        Post process after each iteration.
        """
        pass


class FCFS(Scheduler):
    """A first-come-first-serve scheduler.
    Note: It supports pipeline parallelism. It maintains #pp disjoint batches which
    are in the pipeline under execution.
    Note: The requests are in waiting_queue or the batch_queues, and one request
    can only be in one queue at a time.
    """

    def __init__(self, sched_config: SchedConfig, parallel_config: ParallelConfig, block_manager: BlockManager):
        assert (
            sched_config.policy == "fcfs"
        ), f"can not initialize a FCFS scheduler with policy {sched_config.policy}"
        self.sched_config = sched_config
        # If the current batch is full, the requests will be put into the waiting queue.
        self.waiting_queue = []
        # If one request was in batch_queues before, but swapped out, it will be put into the swapped queue.
        self.swapped_queue = []
        # Since pipeline parallelism is used, there are multiple batches in the system.
        self.cur_index = -1
        self.batch_queues = [
            BatchedRequests() for i in range(parallel_config.pipeline_parallel_size)
        ]
        self.parallel_config = copy.deepcopy(parallel_config)
        self.block_manager = block_manager

    def _check_add_to_cur_batch(self, request: Request) -> bool:
        block_size = self.block_manager.cache_config.block_size
        def get_block_needed(length: int):
            return (length + block_size - 1) // block_size
        
        return (
            len(self.batch_queues[self.cur_index]) < self.sched_config.max_batch_size
        ) and (
            self.batch_queues[self.cur_index].get_num_input_tokens()
            + request.get_num_input_tokens()
            <= self.sched_config.max_tokens_per_batch
        ) and (
            sum([
                get_block_needed(len(req.prompt_token_ids) + req.get_output_len())
                for req in self.batch_queues[self.cur_index].requests + [request]
            ]) <= self.block_manager.max_num_gpu_blocks
        )

    # Requests-related methods
    def add_request(self, request: Request) -> None:
        self.waiting_queue.append(request)

    def abort_request(self, request_id: int) -> None:
        # scan the current batch
        for queue in self.batch_queues:
            for _, request in enumerate(queue.requests):
                if request.request_id == request_id:
                    # This request may be under processed by the model currently,
                    # so it is not safe to delete it from current batch directly.
                    # Mark it as finished will release the resources it holds finally.
                    request.is_finished = True
                    return

        # scan the waiting queue
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return

    def _get_last_stage_batch(self) -> BatchedRequests:
        last_stage_index = (
            self.cur_index - 1
        ) % self.parallel_config.pipeline_parallel_size
        return self.batch_queues[last_stage_index]

    def pop_finished_requests(self) -> List[Request]:
        return self._get_last_stage_batch().pop_finished_requests()

    def get_next_batch(self) -> BatchedRequests:
        self.cur_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size

        block_size = self.block_manager.cache_config.block_size
        def get_block_needed(length: int):
            return (length + block_size - 1) // block_size

        # Check whether the blocks on GPU is enough for the next batch.
        # If not, swap out the last request
        while sum([
            get_block_needed(len(req.prompt_token_ids) + req.get_output_len())
            for req in self.batch_queues[self.cur_index].requests
        ]) > self.block_manager.max_num_gpu_blocks:
            logger.info("No enough GPU blocks. Swap-out triggered")
            request = self.batch_queues[self.cur_index].requests.pop(-1)
            self.swapped_queue.append(request)
            self.block_manager.swap_out_requests([request])

        # Try to add in some new requests. Consider requests in the swapped queue first.
        while len(self.swapped_queue) > 0 or len(self.waiting_queue) > 0:
            if len(self.swapped_queue) > 0:
                request = self.swapped_queue[0]
                if self._check_add_to_cur_batch(request):
                    logger.info("Swap-in triggered")
                    self.block_manager.swap_in_requests([request])
                    self.batch_queues[self.cur_index].add_request(request)
                    self.swapped_queue.pop(0)
                else:
                    break
            else:
                request = self.waiting_queue[0]
                if self._check_add_to_cur_batch(request):
                    self.batch_queues[self.cur_index].add_request(request)
                    self.waiting_queue.pop(0)
                else:
                    break
        return self.batch_queues[self.cur_index]

    # Getter functions
    def get_total_num_requests(self) -> int:
        return self.get_processing_num_requests() + self.get_waiting_num_requests()

    def get_processing_num_requests(self) -> int:
        num = 0
        for batch in self.batch_queues:
            num = num + len(batch.requests)
        return num

    def get_waiting_num_requests(self) -> int:
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        return (
            f"FCFS(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )


class SRPT(Scheduler):
    """Serve prompts in the optimal SRPT manner.

    Note: It does not support pipeline parallelism. It is a artificial scheduler
    which is used to bound the optimal performance for JCT, since it needs the
    output length of each request in advance. Also, it assumes that the request
    will always ignore eos and run max_tokens iterations.

    Note: All the requests are in the requests_queue or cur_batch, and one request
    can only be in one queue at a time.

    """

    def __init__(self, sched_config: SchedConfig):
        assert (
            sched_config.policy == "srpt"
        ), f"can not initialize a SRPT scheduler with policy {sched_config.policy}"
        self.sched_config = sched_config
        self.requests_queue = []
        # In each iteration, SRPT scheduler picks requests from self.requests_queue,
        # push them into self.cur_batch and return to the LLMEngine.
        self.cur_batch = BatchedRequests()

    def _profile_remaining_time(self, request: Request) -> float:
        # FIXME: zili dose not count phrase time here,
        # simulates the time with the number of tokens
        if request.sampling_params.ignore_eos:
            warnings.warn(
                "The request's output len may be unpredicitable.", UserWarning
            )
        return request.sampling_params.max_tokens - request.get_output_len()

    def _pop_request(self):
        requests = self.requests_queue
        retRequest = min(requests, key=lambda p: self._profile_remaining_time(p))
        self.requests_queue.remove(retRequest)
        return retRequest

    def _check_add_to_cur_batch(self, request: Request) -> bool:
        return (
            self.cur_batch.get_num_input_tokens() + request.get_num_input_tokens()
            <= self.sched_config.max_tokens_per_batch
        ) and (len(self.cur_batch) < self.sched_config.max_batch_size)

    def add_request(self, request: Request) -> None:
        self.requests_queue.append(request)

    def abort_request(self, request_id: int) -> None:
        for request in self.cur_batch.requests:
            if request.request_id == request_id:
                # This request may be under processed by the model currently,
                # so it is not safe to delete it from current batch directly.
                # Mark it as finished will release the resources it holds finally.
                request.is_finished = True
                return

        for i, request in enumerate(self.requests_queue):
            if request.request_id == request_id:
                del self.requests_queue[i]
                return

    def get_next_batch(self) -> BatchedRequests:
        # put the requests in current batch back in the queue
        self.requests_queue.extend(self.cur_batch.requests)
        # pick the requests with the shortest remaining time
        self.cur_batch = BatchedRequests()
        while len(self.requests_queue) > 0:
            request = self._pop_request()
            if self._check_add_to_cur_batch(request):
                self.cur_batch.add_request(request)
            else:
                self.requests_queue.append(request)
                break
        return self.cur_batch

    def pop_finished_requests(self) -> List[Request]:
        return self.cur_batch.pop_finished_requests()

    def get_total_num_requests(self) -> int:
        return len(self.requests_queue) + len(self.cur_batch)

    def get_processing_num_requests(self) -> int:
        return len(self.cur_batch)

    def get_waiting_num_requests(self) -> int:
        return len(self.requests_queue)

    def __repr__(self) -> str:
        return (
            f"SRPT(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )


class MLFQ(Scheduler):
    """Serve prompts with skip-join Multi-Level Feedback Queue.

    Note: It supports pipeline parallelism. It maintains #pp disjoint batches which
    are in the pipeline under execution.

    """

    class Priority_Queue:
        def __init__(self, priority: int):
            self.priority = priority
            self.requests = []

        def push_front(self, request) -> None:
            self.requests.insert(0, request)

        def push_back(self, request) -> None:
            self.requests.append(request)

        def pop_front(self):
            return self.requests.pop(0)

        def __len__(self):
            return len(self.requests)

    class Priority_Queues:
        def __init__(self):
            self.queues: List[MLFQ.Priority_Queue] = []

        def add_new_queue(self, priority: int) -> None:
            if priority >= len(self.queues):
                for p in range(len(self.queues), priority + 1):
                    self.queues.append(MLFQ.Priority_Queue(p))

        def pop_front(self) -> None:
            for priority in range(len(self.queues)):
                if len(self.queues[priority]) > 0:
                    return self.queues[priority].pop_front()

        def push_back(self, request) -> None:
            self.add_new_queue(request.get_priority())
            self.queues[request.get_priority()].push_back(request)

        def push_front(self, request) -> None:
            self.add_new_queue(request.get_priority())
            self.queues[request.get_priority()].push_front(request)

        def del_request(self, request_id: int) -> None:
            for queue in self.queues:
                for i, request in enumerate(queue.requests):
                    if request.request_id == request_id:
                        del queue.requests[i]
                        return

        def get_num_requests_in_top_queue(self, num_queues=2) -> int:
            for priority in range(len(self.queues)):
                if len(self.queues[priority]) > 0:
                    num_requests_in_top_queue = 0
                    for i in range(num_queues):
                        if priority + i < len(self.queues):
                            num_requests_in_top_queue += len(self.queues[priority + i])
                    return num_requests_in_top_queue
            return 0

        def __len__(self):
            return sum([len(q) for q in self.queues])

    def __init__(
        self,
        sched_config: SchedConfig,
        parallel_config: ParallelConfig,
        block_manager: BlockManager,
        proactive_offloading: bool = True,
        num_min_free_blocks_threshold: int = 0,
        num_queues_for_prediction: int = 2,
        use_skip_join: bool = True,
    ):
        assert (
            sched_config.policy == "mlfq" or sched_config.policy == "sj-mlfq"
        ), f"can not initialize a MLFQ scheduler with policy {sched_config.policy}"
        self.sched_config = sched_config
        self.parallel_config = copy.deepcopy(parallel_config)
        self.block_manager = block_manager
        self.proactive_offloading = proactive_offloading
        self.num_min_free_blocks_threshold = num_min_free_blocks_threshold
        self.num_queues_for_prediction = num_queues_for_prediction
        self.use_skip_join = use_skip_join
        
        self.iteration_num = 0

        # Load profiling results
        if use_skip_join:
            assert (
                sched_config.profiling_file is not None
            ), "skip-join MLFQ needs profiling results"
        profiling_db = ProfilingDatabase(
            sched_config.profiling_file, new_database=False
        )
        self.profile_res = profiling_db.results[sched_config.model_name]
        print(f"Loaded profiling results for {sched_config.model_name}, {self.profile_res=}")

        # Multi-level Feedback Queue
        self.priority_queues: self.Priority_Queues = self.Priority_Queues()
        # Since pipeline parallelism is used, there may be multiple batches under processing.
        self.cur_index = -1
        self.batch_queues = [
            [] for _ in range(parallel_config.pipeline_parallel_size)
        ]  # List[List[Request]]

        # Just some magic numbers, need to be tuned.
        self.base_quantum = 0.01  # 10 ms
        self.threshold = 2
        
        self.starvation_threshold = 3.  # 3 seconds
        self.starvation_period = 1000  # 1000 iterations

    def _profile_prompt_phrase(self, request: Request) -> float:
        bw = (
            request.sampling_params.best_of
            if request.sampling_params.use_beam_search
            else 1
        )
        batch_size = 1
        input_len = request.get_input_len()
        pp = self.sched_config.parallel_config.pipeline_parallel_size
        tp = self.sched_config.parallel_config.tensor_parallel_size

        latency_list = self.profile_res.get_latency_list(
            pp,
            tp,
            batch_size,
            bw,
            input_len,
        )
        
        if len(latency_list) == 0:
            print(f"Warning: latency_list is empty for {request=}, {pp=}, {tp=}, {batch_size=}, {bw=}, {input_len=}", flush=True)

        return latency_list[0]

    def _profile_decoding_time(self, request: Request) -> float:
        bw = (
            request.sampling_params.best_of
            if request.sampling_params.use_beam_search
            else 1
        )
        batch_size = self.sched_config.max_batch_size
        input_len = request.get_input_len()
        pp = self.sched_config.parallel_config.pipeline_parallel_size
        tp = self.sched_config.parallel_config.tensor_parallel_size

        latency_list = self.profile_res.get_latency_list(
            pp,
            tp,
            batch_size,
            bw,
            input_len,
        )

        return sum(latency_list[1:]) / len(latency_list[1:])

    def add_request(self, request: Request) -> None:
        if self.use_skip_join:
            prompt_time = self._profile_prompt_phrase(request)
            priority = 0
            while pow(self.threshold, priority) * self.base_quantum < prompt_time:
                priority += 1
            request.set_priority(priority)
            self.priority_queues.push_back(request)
        else:
            request.set_priority(0)
            self.priority_queues.push_back(request)

    def abort_request(self, request_id: int) -> None:
        for queue in self.batch_queues:
            for prequest in queue:
                if prequest.request_id == request_id:
                    # This request may be under processed by the model currently,
                    # so it is not safe to delete it from current batch directly.
                    # Mark it as finished will release the resources it holds finally.
                    prequest.request.is_finished = True
                    return
        self.priority_queues.del_request(request_id)

    def pop_finished_requests(self) -> List[Request]:
        last_stage_index = (
            self.cur_index - 1
        ) % self.parallel_config.pipeline_parallel_size

        finished_reqs = []
        for request in self.batch_queues[last_stage_index]:
            if request.is_finished:
                finished_reqs.append(request)
            else:
                # put the request back to mlfq and try to demote it
                if request.get_process_time() > self.base_quantum * pow(
                    self.threshold, request.get_priority()
                ):
                    request.set_priority(request.get_priority() + 1)
                    request.reset_process_time()
                self.priority_queues.push_front(request)
        self.batch_queues[last_stage_index] = []
        return finished_reqs

    def post_process(self) -> None:
        self.reserve_free_blocks([])

    def _check_add_to_cur_batch(self, request) -> bool:
        return (
            sum(
                [
                    preq.get_num_input_tokens() if preq.get_output_len() == 0 else 0
                    for preq in self.batch_queues[self.cur_index]
                ]
            )
            + request.get_num_input_tokens()
            <= self.sched_config.max_tokens_per_batch
        ) and (
            len(self.batch_queues[self.cur_index]) < self.sched_config.max_batch_size
        )

    def get_next_batch(self) -> BatchedRequests:
        self.cur_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size

        assert len(self.batch_queues[self.cur_index]) == 0

        pending_list = []
        while len(self.priority_queues) > 0:
            request = self.priority_queues.pop_front()
            if request.is_context_stage() and self.block_manager.get_num_avail_cpu_blocks() / self.block_manager.max_num_cpu_blocks < 0.1: # TODO: do not use this hard-coded threshold
                pending_list.append(request)
                continue
            elif self._check_add_to_cur_batch(request):
                self.batch_queues[self.cur_index].append(request)
            else:
                # put the priority request back
                self.priority_queues.push_front(request)
                break
        
        for request in pending_list:
            self.priority_queues.push_front(request)
        
        request_list = self.batch_queues[self.cur_index]
        swapped_request_list, execute_request_list = self.reserve_free_blocks(request_list)
        self.batch_queues[self.cur_index] = execute_request_list
        # push back swapped_request_list
        for request in reversed(swapped_request_list):
            self.priority_queues.push_front(request)
            
        self.iteration_num += 1
        
        if self.iteration_num % self.starvation_period == 0:
            self.prevent_starvation()

        return BatchedRequests(execute_request_list)

    # Getter Functions
    def get_total_num_requests(self) -> int:
        return self.get_processing_num_requests() + self.get_waiting_num_requests()

    def get_processing_num_requests(self) -> int:
        return sum([len(queue) for queue in self.batch_queues])

    def get_waiting_num_requests(self) -> int:
        return len(self.priority_queues)

    def __repr__(self) -> str:
        return (
            f"MLFQ(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch}, "
            f"use_skip_join={self.use_skip_join})"
        )

    def predict_num_blocks_needed(self) -> int:
        if self.proactive_offloading:
            num_predicted_blocks_needed = max(
                self.sched_config.max_batch_size,
                self.num_min_free_blocks_threshold,
            )
            return num_predicted_blocks_needed
        else:
            return 0
        
    def prevent_starvation(self) -> None:
        """
        Prevent starvation of the request by promoting it to the top queue.
        """
        promote_reqs = []
        cur_time = time.time()
        
        for q in self.priority_queues.queues:
            buffer = []
            while len(q) > 0:
                request = q.pop_front()
                if cur_time - request.last_step_time >= self.starvation_threshold:
                    promote_reqs.append(request)
                else:
                    buffer.append(request)
            
            for request in buffer:
                q.push_back(request)
        
        # promote the requests in starvation
        for request in promote_reqs:
            request.set_priority(0)
            self.priority_queues.push_front(request)
            
    def set_starvation_threshold(self, threshold: float) -> None:
        self.starvation_threshold = threshold
        
    def set_starvation_period(self, period: int) -> None:
        self.starvation_period = period

    def reserve_free_blocks(self, pinned_requests: List[Request]) -> Tuple[List[Request], List[Request]]:
        """
        Reserve free GPU blocks for pinned requests. If GPU blocks are not enough, swap out low priority requests.
        After that, if GPU blocks are still not enough, swap out pinned requests.
        If proactive_offloading is enabled, it also reserve blocks for predicted incoming requests and swap in
        high priority requests in advance.
        NOTE: pinned_requests is not modified in this function.
        Return: swapped_out pinned requests, executed pinned requests
        """
        # reserve blocks for predicted incoming requests
        num_free_blocks_threshold = self.predict_num_blocks_needed()

        # reserve gpu blocks for pinned requests
        num_blocks_needed = 0
        for request in pinned_requests:
            # if the request is on CPU or not allocated, reserve blocks for it
            location = self.block_manager.get_location(request.request_id)
            if location is None:
                num_blocks_needed += self.block_manager.get_num_blocks_needed(request)
            elif location == BlockLocation.CPU:
                num_blocks_needed += self.block_manager.get_num_blocks_needed(request)
            else:
                num_blocks_needed += self.block_manager.get_num_append_blocks_needed(request)

        num_swap_out_blocks_needed = (
            num_free_blocks_threshold
            + num_blocks_needed
            - self.block_manager.get_num_avail_gpu_blocks()
        )
        swap_out_needed = num_swap_out_blocks_needed > 0

        # the pinned requests we really execute
        execute_pinned_requests = pinned_requests.copy()
        # the pinned requests we put back due to swapped out
        swapped_pinned_requests: List[Request] = []

        # swap out low priority requests if GPU blocks are not enough
        if swap_out_needed:
            # logger.info(
            #     f"threshold: {num_free_blocks_threshold}, pin needed: {num_blocks_needed}, available: {self.block_manager.get_num_avail_gpu_blocks()},swap out needed: {num_swap_out_blocks_needed}"
            # )
            pinned_request_ids = set(
                [request.request_id for request in pinned_requests]
            )
            # swap out from the lowest priority request
            for priority in reversed(range(len(self.priority_queues.queues))):
                for request in reversed(
                    self.priority_queues.queues[priority].requests
                ):
                    # pinned request must have already been popped from MLFQ,
                    assert request.request_id not in pinned_request_ids
                    # running request should not be in MLFQ.
                    assert request.is_running == False
                    if num_swap_out_blocks_needed <= 0:
                        break
                    if (self.block_manager.get_location(request.request_id)== BlockLocation.GPU):
                        num_swap_out_blocks_needed -= (
                            self.block_manager.get_allocated_num_blocks(
                                request.request_id
                            )
                        )
                        # logger.info(f"swap out request {request.request_id}")
                        self.block_manager.swap_out_requests([request])
                if num_swap_out_blocks_needed <= 0:
                    break
            if num_swap_out_blocks_needed > 0:
                # if we still need to swap out blocks, swap out pinned requests
                # location of pinned requests may be in CPU/GPU or none now
                while num_swap_out_blocks_needed > 0 and len(execute_pinned_requests) > 0:
                    request = execute_pinned_requests.pop(-1)
                    swapped_pinned_requests.append(request)
                    num_swap_out_blocks_needed -= self.block_manager.get_num_blocks_needed(request)
                    # logger.info(f"swap out pinned request {request.request_id}")
                    location = self.block_manager.get_location(request.request_id)
                    if location is not None and location == BlockLocation.GPU:
                        self.block_manager.swap_out_requests([request])
            assert num_swap_out_blocks_needed <= 0

        # swap in pinned requests if needed
        for request in execute_pinned_requests:
            if self.block_manager.get_location(request.request_id) == BlockLocation.CPU:
                # logger.info(f"swap in pinned request {request.request_id}")
                self.block_manager.swap_in_requests([request])

        # swap in high priority requests if (1) no swap out gets executed, avoid ping-pong swapping (2) proactive swapping is enabled
        if not swap_out_needed and self.proactive_offloading:
            swap_quata = self.sched_config.max_batch_size
            for priority in range(len(self.priority_queues.queues)):
                if swap_quata <= 0:
                    break
                for request in self.priority_queues.queues[priority].requests:
                    if (
                        self.block_manager.get_location(request.request_id)
                        == BlockLocation.CPU
                    ):
                        # swap in the request if there are enough free blocks
                        if (
                            self.block_manager.get_num_avail_gpu_blocks()
                            >= num_free_blocks_threshold
                            + num_blocks_needed
                            + self.block_manager.get_allocated_num_blocks(
                                request.request_id
                            )
                        ):
                            # logger.info(
                            #     f"proactively swap in request {request.request_id}"
                            # )
                            self.block_manager.swap_in_requests([request])
                    # reduce the quata no matter if the request needs swapping in
                    swap_quata -= 1
                    if swap_quata <= 0:
                        break

        return swapped_pinned_requests, execute_pinned_requests

def get_scheduler(
    sched_config: SchedConfig,
    parallel_config: ParallelConfig,
    block_manager: BlockManager,
) -> Scheduler:
    if sched_config.policy == "fcfs":
        return FCFS(sched_config, parallel_config, block_manager)
    elif sched_config.policy == "srpt":
        return SRPT(sched_config)
    elif "mlfq" in sched_config.policy:
        return MLFQ(
            sched_config,
            parallel_config,
            block_manager,
            proactive_offloading=sched_config.proactive_offloading,
            num_min_free_blocks_threshold=sched_config.num_min_free_blocks_threshold,
            num_queues_for_prediction=sched_config.num_queues_for_prediction,
            use_skip_join=sched_config.use_skip_join,
        )
    else:
        raise NotImplementedError(
            f"scheduler policy {sched_config.policy} is not supported"
        )