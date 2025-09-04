import pytest
import time

from fastserve.request import Request
from fastserve.block_manager import BlockManager
from fastserve.config import ModelConfig, ParallelConfig, CacheConfig


@pytest.mark.parametrize("block_size", [4, 8])
@pytest.mark.parametrize("input_len", [4, 6, 11])
@pytest.mark.parametrize("output_len", [1, 4, 8])
def test_block_manager(block_size, input_len, output_len):
    block_manager = BlockManager(
        64,
        64,
        ModelConfig(
            model="facebook/opt-1.3b",
            tokenizer="facebook/opt-1.3b",
            trust_remote_code=True,
        ),
        ParallelConfig(1, 1),
        CacheConfig(block_size, 1024),
    )
    req0 = Request(time.time(), 0, "To be or not to be,", [1] * input_len)

    block_manager.allocate_blocks(req0)
    num_allocated_blocks = (input_len + block_size - 1) // block_size
    assert (
        block_manager.get_allocated_num_blocks(req0.request_id) == num_allocated_blocks
    )

    assert block_manager.get_partial_block_table([req0.request_id]) == [
        list(range(num_allocated_blocks))
    ]

    for _ in range(output_len):
        req0.add_generated_token("", 1)

    block_manager.allocate_blocks(req0)
    num_allocated_blocks = (input_len + output_len + block_size - 1) // block_size
    assert (
        block_manager.get_allocated_num_blocks(req0.request_id) == num_allocated_blocks
    )
    assert block_manager.get_partial_block_table([req0.request_id]) == [
        list(range(num_allocated_blocks))
    ]

    block_manager.free_blocks(req0.request_id)
    assert block_manager.get_allocated_num_blocks(req0.request_id) == 0
