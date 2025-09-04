import os, sys, dataclasses
from tabulate import tabulate


@dataclasses.dataclass
class Testcase:
    model_name: str
    input_len: int
    batch_size: int
    output_len: int
    tensor_para_size: int


@dataclasses.dataclass
class BenchmarkResult:
    testcase: Testcase
    context_stage_time: float
    decoding_stage_time: float


MULTI_GPU_GPT_EXAMPLE_PATH = (
    "../../reference_proj/fastertransformer_ref/build/bin/multi_gpu_gpt_example"
)

MODEL_NAME_TO_DIR_MAP = {
    "opt-1.3b": "../../reference_proj/fastertransformer_ref/models/opt/opt-1.3b/{tensor_para_size}-gpu",
    "opt-6.7b": "../../reference_proj/fastertransformer_ref/models/opt/opt-6.7b/{tensor_para_size}-gpu",
    "opt-13b": "../../reference_proj/fastertransformer_ref/models/opt/opt-13b/{tensor_para_size}-gpu"
}

TESTCASES = {
    # Order: model_name, input_len, batch_size, output_len (decoding_step+1), tensor_para_size
    # Here output_len should be decoding_step + 1 since in FastTransformer, if
    # we set output_len to x, then the model will generate x - 1 tokens, i.e.
    # decoder->forward() will be called x-1 times.
    # So we set output_len to 17 to generate 16 tokens.
    
    "opt_1.3b_diff_batch_size": [
        Testcase("opt-1.3b", 256, batch_size, 17, 1)
        for batch_size in [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128]
    ],

    "opt_6.7b_diff_batch_size": [
        Testcase("opt-6.7b", 128, batch_size, 17, 1)
        for batch_size in [1, 2, 4, 8, 16, 32, 48, 64, 80, 96]
    ],

    "opt_13b_diff_batch_size": [
        Testcase("opt-13b", 256, batch_size, 17, 1)
        for batch_size in [1, 2, 4, 8, 16, 32, 48, 64]
    ],

    "opt_6.7b_tensor_parallel": [
        Testcase("opt-6.7b", 256, 64, 17, tensor_para_size)
        for tensor_para_size in [1, 2, 4, 8]
    ]
}


def benchmark(testcase: Testcase) -> BenchmarkResult:
    model_name = testcase.model_name
    if model_name not in MODEL_NAME_TO_DIR_MAP:
        print(
            f"Model name {model_name} is not in the model_name_to_dir_map. Quitting..."
        )
        sys.exit(1)
    
    input_len = testcase.input_len
    batch_size = testcase.batch_size
    output_len = testcase.output_len
    tensor_para_size = testcase.tensor_para_size
    model_dir = MODEL_NAME_TO_DIR_MAP[model_name].format(tensor_para_size=tensor_para_size)

    if not os.path.isdir(model_dir):
        print(
            f"{model_dir} does not exist. Please check the path or modify the MODEL_DIR variable in run.py"
        )
        sys.exit(1)

    # Print the benchmark configuration in green background
    print(f"\033[44mBenchmarking with model_name: {model_name}, batch size: {batch_size}, input length: {input_len}, output length: {output_len}, tensor_para_size: {tensor_para_size}\033[0m")

    # Generate gpt_config.ini and write to /tmp/gpt_config.ini
    with open("assets/gpt_config_template.ini", "r") as gpt_config_template_f:
        gpt_config_template = gpt_config_template_f.read()
    gpt_config = gpt_config_template.format(
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        model_dir=model_dir,
        model_name=model_name,
        tensor_para_size=tensor_para_size
    )
    with open("/tmp/gpt_config.ini", "w") as gpt_config_f:
        gpt_config_f.write(gpt_config)

    # Generate input.csv and write to /tmp/input.csv
    with open("assets/input_tokens.txt", "r") as input_tokens_f:
        input_tokens = input_tokens_f.readline().strip().split(" ")[:input_len]
        assert len(input_tokens) == input_len
        input_tokens = [input_tokens for _ in range(batch_size)]
    with open("/tmp/input.csv", "w") as input_f:
        for input_token in input_tokens:
            input_f.write(" ".join(input_token) + "\n")

    # Run the benchmark
    ret_code = os.system(f"mpirun -n {tensor_para_size} \
        {MULTI_GPU_GPT_EXAMPLE_PATH} /tmp/gpt_config.ini /tmp/input.csv \
        > /tmp/benchmark_stdout.txt \
        2> /tmp/benchmark_stderr.txt"
    )
    if ret_code != 0:
        print("Benchmark failed. Quitting...")
        print(
            "You can check /tmp/benchmark_stdout.txt and /tmp/benchmark_stderr.txt for more information."
        )
        sys.exit(1)

    # Read the result
    context_stage_time = decoding_stage_time = -1
    with open("/tmp/benchmark_stderr.txt", "r") as benchmark_stderr_f:
        for line in benchmark_stderr_f.readlines():
            if line.startswith("context_stage_time"):
                context_stage_time = float(line.split()[1].strip())
            elif line.startswith("decoding_stage_time"):
                decoding_stage_time = float(line.split()[1].strip())
    if context_stage_time == -1 or decoding_stage_time == -1:
        print(
            "Benchmark failed since context_stage_time or decoding_stage_time is not found in /tmp/benchmark_stderr.txt. Quitting..."
        )
        print(
            "You can check /tmp/benchmark_stdout.txt and /tmp/benchmark_stderr.txt for more information."
        )
        sys.exit(1)
    print(f"Context stage: {context_stage_time:.2f} ms")
    print(f"Decoding stage: {decoding_stage_time:.2f} ms")
    print(f"Total: {context_stage_time + decoding_stage_time:.2f} ms")

    return BenchmarkResult(testcase, context_stage_time, decoding_stage_time)


def main(exp_name):
    # Check whether MULTI_GPU_GPT_EXAMPLE_PATH exist
    if not os.path.isfile(MULTI_GPU_GPT_EXAMPLE_PATH):
        print(
            f"{MULTI_GPU_GPT_EXAMPLE_PATH} does not exist (or is not a file). Please check the path or modify the MULTI_GPU_GPT_EXAMPLE_PATH variable in run.py"
        )
        sys.exit(1)

    # Check the FMHA_ENABLE environment variable
    use_fmha = False
    if "FMHA_ENABLE" not in os.environ or os.environ["FMHA_ENABLE"] != "ON":
        print(
            "Pay attention: FMHA_ENABLE environment variable is not set to ON. FastTransformer will use unfused multihead attention"
        )
        print(
            "If you want to use fused multihead attention, please set FMHA_ENABLE=ON in your environment, e.g. by running export FMHA_ENABLE=ON in your shell"
        )
        use_fmha = False
    else:
        print(
            "FMHA_ENABLE environment variable is set to ON. FastTransformer will use fused multihead attention"
        )
        use_fmha = True

    # Run the benchmark
    results = []
    for testcase in TESTCASES[exp_name]:
        result = benchmark(testcase)
        results.append(result)

    # Print the results
    print("\033[42mBenchmark results\033[0m")
    print(
        tabulate(
            [
                [
                    result.testcase.model_name,
                    result.testcase.input_len,
                    result.testcase.batch_size,
                    result.testcase.output_len,
                    result.context_stage_time,
                    result.decoding_stage_time,
                ]
                for result in results
            ],
            headers=[
                "model\nname",
                "input\nlen",
                "batch\nsize",
                "output\nlen",
                "context_stage\ntime (ms)",
                "decoding_stage\ntime (ms)",
            ],
        )
    )
    print("Use FHMA: ", use_fmha)
    print()
    print(
        "The following table is formatted with tabs(\\t) for easy copy-paste to Excel."
    )
    print(
        tabulate(
            [
                [
                    result.testcase.model_name,
                    result.testcase.input_len,
                    result.testcase.batch_size,
                    result.testcase.output_len,
                    result.context_stage_time,
                    result.decoding_stage_time,
                ]
                for result in results
            ],
            tablefmt="tsv",
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <experiment_name>")
        print(f"Available experiment names are:")
        for exp_name in TESTCASES.keys():
            print(f"\t{exp_name}")
        sys.exit(1)
    exp_name = sys.argv[1]
    if exp_name not in TESTCASES.keys():
        print(f"Experiment name \"{exp_name}\" not found!")
        print(f"Available experiment names are:")
        for exp_name in TESTCASES.keys():
            print(f"\t{exp_name}")
        sys.exit(1)
    main(exp_name)
