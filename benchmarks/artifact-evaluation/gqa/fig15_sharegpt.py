import lib
from matplotlib import pyplot as plt
import dataclasses
import numpy as np
import os
from lib import ExpResult

os.environ["EXP_RESULT_ROOT"] = "/users/zzl/FastServe/benchmarks/artifact-evaluation/gqa"

def read_result_series(exp_name: str, file_prefix: str) -> list[ExpResult]:
    exp_result_dir = f"{lib.get_exp_result_root()}/{exp_name}"
    file_list = os.listdir(exp_result_dir)
    file_list = [file for file in file_list if file.startswith(file_prefix) and file.endswith("-client-log")]

    results: list[ExpResult] = []
    for file in file_list:
        file_stripped = file[len(file_prefix)+len("-"):-len("-client-log")]
        num_prompts, req_rate = file_stripped.split("-")
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
    tpots = []

    for i in range(len(new_results)):
        req_rates.append(new_results[i].req_rate)
        tpots.append(np.mean([new_results[i].req_results[j].tpot for j in range(len(new_results[i].req_results))]))

    return req_rates, tpots

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

# Set font and figure size
font_size = 33
marker_size = 12
plt.rc('font',**{'size': font_size})
plt.rc('pdf',fonttype = 42)
# fig_size = plt.rcParams['figure.figsize']
fig_size = (12, 4.6)
fig, axes = plt.subplots(nrows=1, ncols=1, sharey=False, figsize=fig_size)
matplotlib.rcParams['xtick.minor.size'] = 4.
matplotlib.rcParams['xtick.major.size'] = 8.
matplotlib.rcParams['ytick.major.size'] = 6.
matplotlib.rcParams['ytick.minor.visible'] = False
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

num_subfigs = 1
num_curves = 3

# line setting
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# colors = {0: '#7bc8f6', 1: '#d1768f'}
labels = {0: 'vLLM', 1:'FastServe-FCFS', 2:'FastServe'}
markers = {0: 'o', 1: '^', 2: 's' , 3: 'd'}

# x-axis setting
x_labels = {0: 'Job Arrival Rate (job/s)', 1: 'Job Arrival Rate (job/s)', 2: 'Job Arrival Rate (job/s)', 3: '(d) NO. of slots'}

# y-axis setting
y_label = 'Latency (s/token)'

_, vllm = read_result_series("logs", "exp-vllm-sharegpt")
_, fastserve_fcfs = read_result_series("logs", "exp-fcfs-sharegpt")
_, fastserve = read_result_series("logs", "exp-fastserve-sharegpt")


rate_y = [vllm, fastserve_fcfs, fastserve]

rate_x_vllm = [1, 2, 3, 3.3, 3.8, 4, 5]
rate_x_fcfs = [1, 2, 3, 3.3, 3.5, 3.8, 4, 5, 5.5]
rate_x_fastserve = [2, 3, 3.3, 3.5, 3.8, 4, 5, 5.5, 6, 7, 8, 9, 10, 11, 12]

rate_x_ticks = [i * 1 for i in range(0, 11, 1)] # [4, 8, 12, 16]
rate_y_ticks = [i * 0.25 for i in range(0, 5, 1)] # [150, 300, 450, 600]

# Plot x ticks and label
for j in range(num_subfigs):
    if j == 0:
        axes.set_xlabel(x_labels[j])
        axes.set_xlim(left=0, right=10)
        # ax1.get_xaxis().get_major_formatter().labelOnlyBase = False
        # axes.set_xscale('log')
        axes.get_xaxis().set_tick_params(direction='in', pad=7)
        axes.get_xaxis().set_tick_params(which='minor', direction='in')
        # axes.set_xticklabels(x_ticklabels)
        axes.set_xticks(rate_x_ticks)
        
# Plot y ticks and label
for j in range(num_subfigs):
    if j == 0:
        axes.set_ylabel(y_label)
        axes.set_ylim(bottom=0, top=1.1)
        axes.set_yticks(rate_y_ticks)
        # axes.set_yticklabels(y_ticklabels)
        axes.get_yaxis().set_tick_params('major', direction='in', pad=4)

# Plot curves
lines = [None for i in range(num_curves)]
for j in range(num_subfigs):
    if j == 0 :
        lines[0], = axes.plot(rate_x_vllm, rate_y[0], label=labels[0], marker = markers[0], color=colors[0], lw=3, markersize=marker_size, linestyle='solid',zorder=3)
        lines[1], = axes.plot(rate_x_fcfs, rate_y[1], label=labels[1], marker = markers[1], color=colors[1], lw=3, markersize=marker_size, linestyle='solid',zorder=3)
        lines[2], = axes.plot(rate_x_fastserve, rate_y[2], label=labels[2], marker = markers[2], color=colors[2], lw=3, markersize=marker_size, linestyle='solid',zorder=3)
        # f_vllm = interp1d(rate_y[0], rate_x_vllm, kind='linear')
        # f_fcfs = interp1d(rate_y[1], rate_x_fcfs, kind='linear')
        # f_fastgen = interp1d(rate_y[2], rate_x_fastserve, kind='linear')

# Plot legend
fig.legend(handles=[lines[0], lines[1], lines[2]], handlelength=2.36, 
           ncol=num_curves+1, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False, prop={'size':font_size})

# Save the figure
file_path = os.path.join(lib.get_exp_result_root(), "fig15_sharegpt.pdf")
plt.savefig(file_path, bbox_inches='tight', transparent=True)
print(f"save to {file_path}")