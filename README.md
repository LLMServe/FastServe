# FastServe
FastServe is a fast, efficient and scalable inference serving system for Large Language Models (LLMs). It serves as an easy-to-use Python front-end and utilizes a high performance LLM inference C++ library [SwiftTransformer](https://github.com/LLMServe/SwiftTransformer).

It is fast with:
- preemptive scheduling
- continuous batching
- custom attention kernels
- C++ model implementation

It is memory efficient with:
- proactive memory swapping
- paged attention kernels

It is scalable with:
- megatron-LM tensor parallelism
- streaming pipeline parallelism

It currently supports:
- OPT (facebook/opt-1.3b, facebook/opt-6.7b, ...)
- LLaMA2 (meta-llama/Llama-2-7b, meta-llama/Llama-2-13b, ...)

## Build && Install

```shell
# git clone the project
git clone git@github.com:LLMServe/FastServe.git && cd FastServe

# setup the fastserve conda environment
conda env create -f environment.yml && conda activate fastserve

# clone and build the SwiftTransformer library  
git clone https://github.com/LLMServe/SwiftTransformer.git && cd SwiftTransformer && git submodule update --init --recursive && cmake -B build && cmake --build build -j$(nproc) && cd ..

# install fastserve
pip install -e .
```

## Artifact Evaluation

See [benchmarks/artifact-evaluation/README.md](./benchmarks/artifact-evaluation/README.md) for detailed instructions.

## Run

### Offline case
```shell
python fastserve/examples/offline.py
```

### Online case
```shell
# launch api server
python -m fastserve.api_server.fastserve_api_server

# launch client
python fastserve/examples/online.py
```

## Contribution
If you want to contribute to the project, please read [contribution.md](./contribution.md).

## Acknowledgement
The architecture design of FastServe is greatly inspired by [vLLM](https://github.com/vllm-project/vllm).

## Citation
If you use FastServe for your research, please cite our [paper](https://arxiv.org/abs/2305.05920):
```
@misc{wu2023fast,
      title={Fast Distributed Inference Serving for Large Language Models}, 
      author={Bingyang Wu and Yinmin Zhong and Zili Zhang and Gang Huang and Xuanzhe Liu and Xin Jin},
      year={2023},
      eprint={2305.05920},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```