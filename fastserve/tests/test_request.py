import pytest

from fastserve.request import BatchedRequests
from fastserve.tests.utils import get_fake_request


@pytest.mark.parametrize("input_len", [1, 2, 5, 8])
@pytest.mark.parametrize("output_len", [1, 2, 5, 8])
def test_request(input_len, output_len):
    req = get_fake_request(0, input_len, output_len)
    assert req.is_context_stage() == True
    assert req.get_input_len() == input_len
    assert req.get_output_len() == 0
    assert req.get_num_input_tokens() == input_len
    req.add_generated_token("a", 1)
    assert req.is_context_stage() == False
    assert req.get_input_len() == input_len
    assert req.get_output_len() == 1
    assert req.get_num_input_tokens() == 1
    for i in range(2, output_len + 1):
        req.add_generated_token("a", 1)
        assert req.is_context_stage() == False
        assert req.get_input_len() == input_len
        assert req.get_output_len() == i
        assert req.get_num_input_tokens() == 1
    assert req.is_finished == True


def test_batched_request():
    pass
