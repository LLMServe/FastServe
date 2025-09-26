import lib
from matplotlib import pyplot as plt
import dataclasses
import numpy as np
import os
from lib import ExpResult

os.environ["EXP_RESULT_ROOT"] = "/users/zzl/FastServe/benchmarks/artifact-evaluation/large_bs"

def read_result_series(exp_name: str, file_prefix: str, file_suffix: str) -> list[ExpResult]:
    exp_result_dir = f"{lib.get_exp_result_root()}/{exp_name}"
    file_list = os.listdir(exp_result_dir)
    file_list = [file for file in file_list if file.startswith(file_prefix) and file.endswith(file_suffix)]

    results: list[ExpResult] = []
    for file in file_list:
        file_stripped = file[len(file_prefix)+len("-"):-len("-client-log")]
        num_prompts, req_rate, _ = file_stripped.split("-")
        num_prompts = int(num_prompts)
        req_rate = float(req_rate)
        req_results = lib.ReqResult.from_client_log(exp_result_dir + "/" + file)
        if len(req_results) == 0:
            continue
        results.append(ExpResult(num_prompts, req_rate, req_results))
	
    new_results = []
    for item in results:
        invalid = False
        for item2 in results:
            if item2.req_rate <= item.req_rate and item2.num_prompts > item.num_prompts:
                invalid = True
                break
        if not invalid:
            new_results.append(item)
    new_results.sort(key=lambda x: x.req_rate)

    req_rates = []
    latency_list = []

    for i in range(len(new_results)):
        req_rates.append(new_results[i].req_rate)
        latency_list.append(np.mean([new_results[i].req_results[j].per_token_latency for j in range(len(new_results[i].req_results))]))

    return req_rates, latency_list

import os
import sys
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import palettable
import random
from scipy.interpolate import interp1d

sysname = 'FastServe'

num_subfigs = 1
num_curves = 3

# Set font and figure size
font_size = 15
marker_size = 4
plt.rc('font',**{'size': font_size, 'family': 'Arial'})
plt.rc('pdf',fonttype = 42)
fig_size = (3.5, 2)
fig, axes = plt.subplots(nrows=1, ncols=num_subfigs, sharey=False, figsize=fig_size)
matplotlib.rcParams['xtick.minor.size'] = 4.
matplotlib.rcParams['xtick.major.size'] = 8.
matplotlib.rcParams['ytick.major.size'] = 6.
matplotlib.rcParams['ytick.minor.visible'] = False
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

# line setting
colors = ['C1', 'C2', 'C3', 'C4']
labels = {0: 'vLLM', 1:sysname+'-FCFS', 2:sysname}
markers = {0: '^', 1: 's' , 2: 'd'}

# x-axis setting
x_labels = {0: 'Job Arrival Rate', 1: 'Job Arrival Rate'}

# y-axis setting
y_label = 'Latency (s/token)'

xs = [
    # Figure 1
    [
        # vLLM
        read_result_series("logs", "exp-vllm-sharegpt", "bs=64-client-log")[0],
        # FastGen-FCFS
        read_result_series("logs", "exp-fcfs-sharegpt", "bs=64-client-log")[0],
        # FastGen
        read_result_series("logs", "exp-fastserve-sharegpt", "bs=64-client-log")[0],
    ],
]
ys = [
    # Figure 1
    [
        # vLLM
        read_result_series("logs", "exp-vllm-sharegpt", "bs=64-client-log")[1],
        # FastGen-FCFS
        read_result_series("logs", "exp-fcfs-sharegpt", "bs=64-client-log")[1],
        # FastGen
        read_result_series("logs", "exp-fastserve-sharegpt", "bs=64-client-log")[1],
    ],
]
y_limits = [0.2, 0.5]
x_ticks = [i * 0.25 for i in range(4, 17, 2)] # [4, 8, 12, 16]
y_ticks_list = [
    [i * 0.025 for i in range(0, 10, 2)],
    [i * 0.050 for i in range(0, 11, 2)],
]

handles = [None for i in range(num_curves)]
for j in range(num_subfigs):
    ax: plt.Axes = axes[j] if num_subfigs > 1 else axes
    
    # Plot x ticks and label
    ax.set_xlabel(x_labels[j])
    ax.set_xlim(left=1.0, right=1.2)
    ax.get_xaxis().set_tick_params(direction='in', pad=7)
    ax.get_xaxis().set_tick_params(which='minor', direction='in')
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
        
    # Plot y ticks and label
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0, top=y_limits[j])
    ax.set_yticks(y_ticks_list[j])
    ax.get_yaxis().set_tick_params('major', direction='in', pad=4)
    ax.get_yaxis().set_tick_params(direction='in', pad=4)

    # Plot curves
    f_list: list = []
    for i in range(num_curves):
        print(f"Plotting curve {i} in subplot {j}, xs: {xs[j][i]}, ys: {ys[j][i]}")
        handles[i], = ax.plot(xs[j][i], ys[j][i], label=labels[i], marker=markers[i], color=colors[i], markersize=marker_size, linestyle='solid', zorder=3)
        
    #     f_inter = interp1d(ys[j][i], xs[j][i], kind='linear')
    #     f_inter_val = f_inter(0.25)
    #     f_list.append(f_inter_val)
    # print(f_list)
    # print(f_list[2] / f_list[0], f_list[2] / f_list[1])
    
    # Plot grid
fig.text(0.5, -0.27, 'BS=64.', fontsize=17, fontname='Times New Roman', 
        color='black', ha='center', va='bottom')
# fig.text(0.73, -0.27, '(b) BS=128.', fontsize=17, fontname='Times New Roman',  
#         color='black', ha='center', va='bottom')
# Plot legend
fig.legend(
        handles=handles,
        handlelength=2.36, 
        ncol=num_curves,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        prop={'size':font_size},
        columnspacing=0.9
)

# Save the figure
file_path = './large_bs=64.pdf'
plt.savefig(file_path, bbox_inches='tight', transparent=True)