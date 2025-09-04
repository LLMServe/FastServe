from fastserve.request import Request, SamplingParams


def get_fake_request(request_id, input_len, output_len):
    """Return a fake request which needs fixed output_len iterations."""
    return Request(
        0,
        request_id,
        "<s>" * input_len,
        [0] * input_len,
        SamplingParams(max_tokens=output_len, ignore_eos=True),
    )
