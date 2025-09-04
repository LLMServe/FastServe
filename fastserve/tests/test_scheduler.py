import pytest

from fastserve.config import SchedConfig, ParallelConfig
from fastserve.scheduler import FCFS
from fastserve.tests.utils import get_fake_request


@pytest.mark.parametrize("max_batch_size", [1, 2, 5, 8])
@pytest.mark.parametrize("max_tokens_per_batch", [2, 4, 8, 10])
def test_fcfs(max_batch_size, max_tokens_per_batch):
    sched_policy = SchedConfig("fcfs", max_batch_size, max_tokens_per_batch)
    scheduler = FCFS(sched_policy, ParallelConfig())

    # test max_batch_size
    scheduler.add_request(get_fake_request(0, 1, 1))
    next_batch = scheduler.get_next_batch()
    assert len(next_batch) == 1
    for i in range(1, max_batch_size):
        scheduler.add_request(get_fake_request(i, 1, 1))
    next_batch = scheduler.get_next_batch()
    assert len(next_batch) == min(max_batch_size, max_tokens_per_batch)
    scheduler.add_request(get_fake_request(max_batch_size, 1, 1))
    next_batch = scheduler.get_next_batch()
    assert len(next_batch) == min(max_batch_size, max_tokens_per_batch)

    # clear the batch
    next_batch.requests = []
    assert len(next_batch) == 0

    # test max_tokens_per_batch
    for i in range(max_tokens_per_batch):
        scheduler.add_request(get_fake_request(i, 1, 1))
    next_batch = scheduler.get_next_batch()
    assert len(next_batch) == min(max_batch_size, max_tokens_per_batch)
    scheduler.add_request(get_fake_request(max_tokens_per_batch, 1, 1))
    next_batch = scheduler.get_next_batch()
    assert len(next_batch) == min(max_batch_size, max_tokens_per_batch)
    
def all_add_request(request, schedulers):
    for scheduler in schedulers:
        scheduler.add_request(request)

@pytest.mark.parametrize("max_batch_size", [1, 2, 5, 8])
@pytest.mark.parametrize("max_tokens_per_batch", [2, 4, 8, 10])
@pytest.mark.parametrize("pipeline_size", [2, 4, 8])
def test_fcfs_para(max_batch_size, max_tokens_per_batch, pipeline_size):
    sched_policy = SchedConfig("fcfs", max_batch_size, max_tokens_per_batch)
    schedulers = []
    for i in range(pipeline_size):
        parallel_config = ParallelConfig(1, 0, pipeline_size, i)
        schedulers.append(FCFS(sched_policy, parallel_config))
    
    size = min(max_batch_size, max_tokens_per_batch)
    
    for i in range(size * (pipeline_size-1) + 1):
        all_add_request(get_fake_request(i, 1, 1), schedulers)

    for i in range(pipeline_size):
        # run pipeline_size-1 times forword
        for j in range(pipeline_size-1-i):
            schedulers[i].get_next_batch()
    
    for i in range(pipeline_size):
        next_batch = schedulers[i].get_next_batch()
        if i == 0:
            assert len(next_batch) == 1
            assert next_batch.requests[0].request_id == size * (pipeline_size-1)
        else:
            assert len(next_batch) == size
            for j in range(size):
                assert next_batch.requests[j].request_id == size * (pipeline_size-1-i) + j
    
    # add one more request
    all_add_request(get_fake_request(size * (pipeline_size-1) + 1, 1, 1), schedulers)
    for i in range(pipeline_size):
        for j in range(pipeline_size+2):
            schedulers[i].get_next_batch()
    for i in range(pipeline_size):
        assert (len(schedulers[i].batch_queue[pipeline_size - 1])) == 2 or max_batch_size == 1
    