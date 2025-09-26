# Artifact Evaluation
This is the artifact evaluation of NSDI'26 paper "Iteration-Level Preemptive Scheduling for Large Language Model Inference". The experiments in this artifact are designed for a single-GPU setup.

## Env
After you `ssh` into machine (e.g., xinjin1), run the following command to activate the conda environment.
```
source ~/.bashrc
```

### Figure 11 & Figure 12 & Figure 13 & Figure 18
First, navigate into the `overall` directory which contains the necessary shell scripts. Next, execute all the `overall_*.sh` scripts to run the experiments. These scripts will generate the log data required for the figures. (Note: You can modify the output directory variable inside each `.sh` script to specify where you want to save the log files.) Once all the benchmark scripts have successfully completed, run the `fig*.py` Python scripts to process the logs and generate the final figures. (Note: Ensure the log directory path in the Python scripts matches the location where the benchmark logs were saved.)

For sharegpt dataset, each script may consume 3-5 hours. For alpaca, each script may consume 1-3 hours.

```
cd overall
bash overall_fastserve_sharegpt.sh
bash overall_vllm_sharegpt.sh
bash overall_fcfs_sharegpt.sh
bash overall_cp_sharegpt.sh
bash overall_fastserve_alpaca.sh
bash overall_vllm_alpaca.sh
bash overall_fcfs_alpaca.sh
bash overall_cp_alpaca.sh

# Fig 11
python fig11_sharegpt.py
python fig11_alpaca.py

# Fig 12
python fig12.py

# Fig 13
python fig13.py

# Fig 18
python fig18_sharegpt.py
python fig18_alpaca.py

```

### Figure 14 (~3 hours)
First, navigate into the `large_bs` directory which contains the necessary shell scripts. Next, execute all the `large_bs_*.sh` scripts to run the experiments. These scripts will generate the log data required for the figures. (Note: You can modify the output directory variable inside each `.sh` script to specify where you want to save the log files.) Once all the benchmark scripts have successfully completed, run the `fig*.py` Python scripts to process the logs and generate the final figures. (Note: Ensure the log directory path in the Python scripts matches the location where the benchmark logs were saved.)
```
source ~/.bashrc
conda activate fastserve

cd benchmarks/artifact-evaluation/large_bs

bash large_bs_fastserve_bs=64.sh
bash large_bs_fcfs_bs=64.sh
bash large_bs_vllm_bs=64.sh

python fig14_bs=64.py  # see fig14_bs=64.pdf

bash large_bs_fastserve_bs=128.sh
bash large_bs_fcfs_bs=128.sh
bash large_bs_vllm_bs=128.sh

python fig14_bs=128.py  # see fig14_bs=128.pdf
```

### Figure 15
Each script takes around 1h.
```
bash gqa_fastserve_sharegpt.sh
bash gqa_fcfs_sharegpt.sh
bash gqa_vllm_sharegpt.sh

# Figure 15(a) ShareGPT
python fig15_sharegpt.py  # see fig15_sharegpt.pdf

bash gqa_fastserve_alpaca.sh
bash gqa_fcfs_alpaca.sh
bash gqa_vllm_alpaca.sh

# Figure 15(b) Alpaca
python fig15_alpaca.py  # see fig15_alpaca.pdf
```
