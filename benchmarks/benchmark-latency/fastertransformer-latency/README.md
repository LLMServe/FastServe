# FasterTransformer Latency Benchmark

## Description

This experiment runs Fast Transformer on different `input_length`, `batch_size`, and `output_lenth`s (all input sequences are the same), and then measures its time consumption on context stage and decoding stage. You can customize those parameters in `run.py`.

## Setup

- Download and convert the OPT-1.3B, OPT-6.7B and OPT-30B model.

	You may refer to https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md#run-meta-opt. 

	Generally speaking, you should download the model from HuggingFace by `git lfs clone`.
	(If you think the model is too big you can only download those metadata files and PyTorch model files.)
	Then you should use `reference_proj/fastertransformer_ref/examples/pytorch/gpt/utils/huggingface_opt_convert.py` to convert the model to the format that FastTransformer can use.

	After this step you should have `model.XXX` under `reference_proj/fastertransformer_ref/models/opt/opt-{1.3,6.7,30}b/{1,2,4,8}-gpu`.

- Compile FasterTransformer.

	Do not forget to add `-DCMAKE_BUILD_TYPE=Release` and `-DBUILD_MULTI_GPU=ON` when running `cmake`.

	After this step you should have `multi_gpu_gpt_example` under `reference_proj/fastertransformer_ref/build/bin`

## Launch

`python3 run.py <experiment_name>.`

If you want to run FastTransformer with Fused Multihead Attention enabled, please `export FMHA_ENABLE=ON` before running. Note. Fused Multihead Attention will be dispatched to FlashAttention only when head_size > 64 or max_seq_len > 128.
