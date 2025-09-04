from fastserve import SamplingParams, OfflineLLM

from transformers import AutoTokenizer, OPTForCausalLM, LlamaForCausalLM
import ray

import pytest
from typing import List

class llm_struct:
    def __init__(self) -> None:
        self.input_ids = None
        self.generate_ids = None
        self.answer = None
    
    def __str__(self) -> str:
        return f"input_ids: {self.input_ids}\ngenerate_ids: {self.generate_ids}\nanswer: {self.answer}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def is_same(self, other):
        return self.input_ids == other.input_ids and self.generate_ids == other.generate_ids and self.answer == other.answer


def run_huggingface(test_model: str, prompt: List[str], sampling: SamplingParams) -> llm_struct:
    ret_struct = llm_struct()
    
    # choose model my name
    if test_model.startswith("meta-llama"):
        model = LlamaForCausalLM.from_pretrained(test_model).to("cuda")
    elif test_model.startswith("facebook/opt"):
        model = OPTForCausalLM.from_pretrained(test_model).to("cuda")
    else:
        raise NotImplementedError(f"model {test_model} not supported")
    tokenizer = AutoTokenizer.from_pretrained(test_model)

    inputs = tokenizer(prompt, return_tensors="pt")
    ret_struct.input_ids = inputs.input_ids

    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_length=128)
    ret_struct.generate_ids = generate_ids

    answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    ret_struct.answer = answer
    return ret_struct

def run_fastserve(test_model: str, prompt: List[str], sampling: SamplingParams, tensor_parallel_size:int, pipeline_parallel_size:int) -> llm_struct:
    ret_struct = llm_struct()
    tokenizer = AutoTokenizer.from_pretrained(test_model)
    model = OfflineLLM(model=test_model, tokenizer=tokenizer, SamplingParams=sampling, tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    # convert tensor to List[List[int]]
    inputs.input_ids = inputs.input_ids.tolist()
    ret_struct.input_ids = inputs.input_ids
    
    ret_struct.generate_ids = model.generate(prompt_token_ids=inputs.input_ids, SamplingParams=sampling)
    answer = tokenizer.batch_decode(ret_struct.generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    ret_struct.answer = answer
    
    return ret_struct

@pytest.mark.parametrize("test_model", ["meta-llama/Llama-2-7b", "facebook/opt-1.3b"])
def test_project(test_model):
    prompt = [
        "To be or not to be",
        "The quick brown fox jumps over ",
        "Jin Xin from PKU is",
        "The capital of France is",
    ]
    
    sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128, stop=["\n"])
    
    std_ans = run_huggingface(test_model, prompt, sampling)
    fast_ans = run_fastserve(test_model, prompt, sampling, tensor_parallel_size=1, pipeline_parallel_size=1)
    
    assert std_ans.is_same(fast_ans)
    
    ray.init()
    # get cluster GPU resources
    num_gpus = ray.cluster_resources().get("GPU", 0)
    
    if num_gpus >= 2:
        tp_ans = run_fastserve(test_model, prompt, sampling, tensor_parallel_size=2, pipeline_parallel_size=1)
        pp_ans = run_fastserve(test_model, prompt, sampling, tensor_parallel_size=1, pipeline_parallel_size=2)
        assert tp_ans.is_same(std_ans)
        assert pp_ans.is_same(std_ans)