import argparse
from fastserve import OfflineLLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='The model to use', default='facebook/opt-125m')
args = parser.parse_args()

# Sample prompts.
prompts = [
    "Life blooms like a flower. Far away or by the road. Waiting",
    "A quick brown fox",
    "Artificial intelligence is",
    "To be or not to be,",
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0, top_p=1.0, max_tokens=16, stop=[]
)

# Create an LLM for offline inference.
llm = OfflineLLM(
    model=args.model,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)

# Generate texts from the prompts. The output is a list of Request objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.get_response()
    output_len = output.get_output_len()
    print(
        f"Request {output.request_id}, Prompt: {prompt!r}, Generated text: {generated_text!r} ({output_len} tokens generated)."
    )
