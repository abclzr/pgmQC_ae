import argparse
import os
import sys
import subprocess
import argparse
import pdb
import os
import cotengra as ctg
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Function to validate directory paths
def validate_directory(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Directory {path} does not exist.")
    return path

# get the metric from the metric file. The text in the metric file looks like this:
# Memory usage on cpu: 2752 bytes
# Sparse tensor contraction time taken: 1.33121 ms
# Total flops: 10
def get_metric(path):
    # Read the content of the file
    with open(path, 'r') as file:
        content = file.read()
    # Extract metrics using regular expressions
    memory_usage = re.search(r'Memory usage on cpu: (\d+) bytes', content)
    contraction_time = re.search(r'Sparse tensor contraction time taken: ([\d.]+) ms', content)
    total_flops = re.search(r'Total flops: (\d+)', content)

    # Convert extracted values to appropriate types
    memory_usage = int(memory_usage.group(1)) if memory_usage else None
    contraction_time = float(contraction_time.group(1)) if contraction_time else None
    total_flops = int(total_flops.group(1)) if total_flops else None

    # Print the extracted metrics
    print(f'Memory Usage on CPU: {memory_usage} bytes')
    print(f'Sparse Tensor Contraction Time Taken: {contraction_time} ms')
    print(f'Total FLOPS: {total_flops}')
    return memory_usage, contraction_time, total_flops

# Sparsity of the tensor: 216 / 4096
def get_sparsity_metric(path):
    # Read the content of the file
    with open(path, 'r') as file:
        content = file.read()
    # Extract metrics using regular expressions
    sparsity = re.search(r'Sparsity of the tensor: (\d+) / (\d+)', content)

    assert sparsity != None
    # Print the extracted metrics
    print(f'Sparsity of the tensor: {sparsity.group(1)} / {sparsity.group(2)}')
    return sparsity.group(1), sparsity.group(2)

def plot_metrics(n_qubits_range, data1, data2, ylabel, title, filename, logscale=False, show_reduction=False):
    x = range(len(data1))

    # Plotting the data
    plt.plot(x, data1, label='Ours', marker='o')
    plt.plot(x, data2, label='cuquantum', marker='x')
    
    if show_reduction:
        # Calculate the reduction at the last point
        last_idx = len(data1) - 1
        reduction = -(data2[last_idx] - data1[last_idx]) / data2[last_idx] * 100  # Reduction percentage

        # Annotate the reduction at the last point, offset to avoid overlap with arrow
        plt.annotate(f'{reduction:.1f}%', (last_idx, data1[last_idx]), textcoords="offset points", xytext=(-40,20), ha='center', fontsize=20, fontweight='bold', color='red')
        
        # Add an arrow pointing from the last point of cuquantum to the last point of sparse
        plt.annotate('', xy=(last_idx, data1[last_idx]), xytext=(last_idx, data2[last_idx]),
                    arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=1", color='red', lw=2))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e9:.0f}G' if x >= 1e9 else f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'))

    plt.xticks(x, n_qubits_range, fontsize=20)
    plt.yticks(fontsize=20)
    # Adding labels and title
    plt.xlabel('#Qubits', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    # plt.title(title)
    if logscale:
        plt.yscale('log')

    # Adding a legend
    plt.tight_layout()
    plt.legend(fontsize=22)
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_2metrics(n_qubits_range, data1, data2, data3, ylabel, title, filename, logscale=False, show_reduction=False):
    x = range(len(data1))

    # Plotting the data with labels
    plt.plot(x, data1, label='Ours: 50% pruned', marker='o', color='#1f77b4')
    plt.plot(x, data2, label='Ours: 90% pruned', marker='s', color='#2ca02c')
    plt.plot(x, data3, label='cuquantum', marker='x', color='#ff7f0e')

    if show_reduction:
        # Calculate the reduction between 50% pruned and cuquantum at the last point
        last_idx = len(data3) - 1
        reduction_50 = -(data3[last_idx] - data1[last_idx]) / data3[last_idx] * 100  # Reduction percentage for 50%
        reduction_90 = -(data3[last_idx] - data2[last_idx]) / data3[last_idx] * 100  # Reduction percentage for 90%

        # Annotate the 50% pruned reduction at the last point with a left-curving arrow
        plt.annotate(f'{reduction_50:.1f}%', (last_idx, data1[last_idx]), 
                     textcoords="offset points", xytext=(-40, 20), ha='center', fontsize=16, 
                     fontweight='bold', color='red')
        
        plt.annotate('', xy=(last_idx, data1[last_idx]), xytext=(last_idx, data3[last_idx]),
                     arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=1", color='red', 
                                     lw=2, connectionstyle="arc3,rad=0.3"))  # Left curve

        # Annotate the 90% pruned reduction at the last point with a right-curving arrow
        plt.annotate(f'{reduction_90:.1f}%' if reduction_90 > -99.9 else f'{reduction_90:.2f}%', (last_idx, data2[last_idx]), 
                     textcoords="offset points", xytext=(40, 20), ha='center', fontsize=16, 
                     fontweight='bold', color='red')
        
        plt.annotate('', xy=(last_idx, data2[last_idx]), xytext=(last_idx, data3[last_idx]),
                     arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=1", color='red', 
                                     lw=2, connectionstyle="arc3,rad=-0.3"))  # Right curve
        # Format y-axis in millions (M) and billions (G)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e9:.0f}G' if x >= 1e9 else f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'))

    plt.xticks(x, n_qubits_range, fontsize=20)
    plt.yticks(fontsize=20)
    # Adding labels and title
    plt.xlabel('#Qubits', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    # plt.title(title)
    if logscale:
        plt.yscale('log')

    # Adding a legend
    plt.tight_layout()
    plt.legend(fontsize=22)
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_sparsity(n_qubits_range, data, title, filename, logscale=False):
    x = range(len(data))
    # Convert data to percentages
    data_percentage = [d * 100 for d in data]

    # Plotting the data
    plt.plot(x, data_percentage, label='sparse', marker='o', color='#1f77b4')

    plt.xticks(x, n_qubits_range, fontsize=20)
    plt.yticks(fontsize=20)
    # Adding labels and title
    plt.xlabel('#Qubits', fontsize=22)
    plt.ylabel('Sparsity (%)', fontsize=22)
    # plt.title(title, fontsize=18)
    if logscale:
        plt.yscale('log')
    # Adding percentage formatter for the y-axis
    plt.ylim(70, 100)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Adding a legend
    # plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
def plot_2sparsity(n_qubits_range, data1, data2, title, filename, logscale=False):
    x = range(len(data1))
    # Convert data to percentages
    data_percentage1 = [d * 100 for d in data1]
    data_percentage2 = [d * 100 for d in data2]

    # Plotting the data
    plt.plot(x, data_percentage1, label='50% pruned', marker='o', color='#1f77b4')
    plt.plot(x, data_percentage2, label='90% pruned', marker='o', color='#2ca02c')

    plt.xticks(x, n_qubits_range, fontsize=20)
    plt.yticks(fontsize=20)
    # Adding labels and title
    plt.xlabel('#Qubits', fontsize=22)
    plt.ylabel('Sparsity (%)', fontsize=22)
    # plt.title(title, fontsize=18)
    if logscale:
        plt.yscale('log')
    # Adding percentage formatter for the y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%' if y > 1 else f'{y:.2f}%'))

    # Adding a legend
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_task(task_name, n_qubits_range):
    sp_memory_list = []
    sp_latency_list = []
    sp_flops_list = []
    sparsity_subcircuit2_list = []
    cq_memory_list = []
    cq_latency_list = []
    cq_flops_list = []
    for n in n_qubits_range:
        dir = f'../dataset/{task_name}_{n}'
        sp_memory, sp_latency, sp_flops = get_metric(os.path.join(dir, 'sp_metric.txt'))
        cq_memory, cq_latency, cq_flops = get_metric(os.path.join(dir, 'cuquantum_metric.txt'))
        print(f'For {n} qubits:')
        print(f'  Sparse: Memory: {sp_memory} bytes, Latency: {sp_latency} ms, FLOPS: {sp_flops}')
        print(f'  CuQuantum: Memory: {cq_memory} bytes, Latency: {cq_latency} ms, FLOPS: {cq_flops}')
        subpaths = os.listdir(f'../dataset/{task_name}_{n}/')
        subpaths = [subpath for subpath in subpaths if subpath.startswith('subcircuit')]
        dirs = [f'../dataset/{task_name}_{n}/{subpath}' for subpath in subpaths]
        # dirs = [f'../dataset/{task_name}_{n}_subgraph{i}' for i in [1,2]] # for qft
        sum_sp_value1 = 0
        sum_sp_value2 = 0
        for dir in dirs:
            sp_value1, sp_value2 = get_sparsity_metric(os.path.join(dir, 'sp_metric.txt'))
            sum_sp_value1 += int(sp_value1)
            sum_sp_value2 += int(sp_value2)
        # sp_value1, sp_value2 = get_sparsity_metric(os.path.join(f'../dataset/{task_name}_{n}_subgraph2', 'sp_metric.txt'))
        sp_value = 1. - int(sum_sp_value1) / int(sum_sp_value2)
        sp_memory_list.append(sp_memory)
        sp_latency_list.append(sp_latency)
        sp_flops_list.append(sp_flops)
        cq_memory_list.append(cq_memory)
        cq_latency_list.append(cq_latency)
        cq_flops_list.append(cq_flops)
        sparsity_subcircuit2_list.append(sp_value)

    # Plot the metrics
    plot_metrics(n_qubits_range, sp_memory_list, cq_memory_list, 'Memory (bytes)', task_name, 'figure/' + task_name + "_memory.pdf", logscale=False, show_reduction=True)
    plot_metrics(n_qubits_range, sp_latency_list, cq_latency_list, 'Latency (ms)', task_name, 'figure/' + task_name + "_latency.pdf", logscale=False, show_reduction=False)
    plot_metrics(n_qubits_range, sp_flops_list, cq_flops_list, 'FLOP Count', task_name, 'figure/' + task_name + "_flops.pdf", logscale=False, show_reduction=False)
    
    plot_sparsity(n_qubits_range, sparsity_subcircuit2_list, task_name, 'figure/' + task_name + "_sparsity.pdf")


def plot_2task(task_name1, task_name2, n_qubits_range, title):
    sp_memory_list1 = []
    sp_latency_list1 = []
    sp_flops_list1 = []
    sparsity_list1 = []
    sp_memory_list2 = []
    sp_latency_list2 = []
    sp_flops_list2 = []
    sparsity_list2 = []
    cq_memory_list = []
    cq_latency_list = []
    cq_flops_list = []
    for n in n_qubits_range:
        dir = f'../dataset/{task_name1}_{n}'
        sp_memory1, sp_latency1, sp_flops1 = get_metric(os.path.join(dir, 'sp_metric.txt'))
        cq_memory, cq_latency, cq_flops = get_metric(os.path.join(dir, 'cuquantum_metric.txt'))
        print(f'For {n} qubits:')
        print(f'  Sparse: Memory: {sp_memory1} bytes, Latency: {sp_latency1} ms, FLOPS: {sp_flops1}')
        print(f'  CuQuantum: Memory: {cq_memory} bytes, Latency: {cq_latency} ms, FLOPS: {cq_flops}')
        subpaths = os.listdir(f'../dataset/{task_name1}_{n}/')
        subpaths = [subpath for subpath in subpaths if subpath.startswith('subcircuit')]
        dirs = [f'../dataset/{task_name1}_{n}/{subpath}' for subpath in subpaths]
        sum_sp_value1 = 0
        sum_sp_value2 = 0
        for dir in dirs:
            sp_value1, sp_value2 = get_sparsity_metric(os.path.join(dir, 'sp_metric.txt'))
            sum_sp_value1 += int(sp_value1)
            sum_sp_value2 += int(sp_value2)
        # sp_value1, sp_value2 = get_sparsity_metric(os.path.join(f'../dataset/{task_name}_{n}_subgraph2', 'sp_metric.txt'))
        sp_value = 1. - int(sum_sp_value1) / int(sum_sp_value2)
        sp_memory_list1.append(sp_memory1)
        sp_latency_list1.append(sp_latency1)
        sp_flops_list1.append(sp_flops1)
        cq_memory_list.append(cq_memory)
        cq_latency_list.append(cq_latency)
        cq_flops_list.append(cq_flops)
        sparsity_list1.append(sp_value)
    for n in n_qubits_range:
        dir = f'../dataset/{task_name2}_{n}'
        sp_memory2, sp_latency2, sp_flops2 = get_metric(os.path.join(dir, 'sp_metric.txt'))
        print(f'For {n} qubits:')
        print(f'  Sparse: Memory: {sp_memory2} bytes, Latency: {sp_latency2} ms, FLOPS: {sp_flops2}')
        subpaths = os.listdir(f'../dataset/{task_name2}_{n}/')
        subpaths = [subpath for subpath in subpaths if subpath.startswith('subcircuit')]
        dirs = [f'../dataset/{task_name2}_{n}/{subpath}' for subpath in subpaths]
        sum_sp_value1 = 0
        sum_sp_value2 = 0
        for dir in dirs:
            sp_value1, sp_value2 = get_sparsity_metric(os.path.join(dir, 'sp_metric.txt'))
            sum_sp_value1 += int(sp_value1)
            sum_sp_value2 += int(sp_value2)
        # sp_value1, sp_value2 = get_sparsity_metric(os.path.join(f'../dataset/{task_name}_{n}_subgraph2', 'sp_metric.txt'))
        sp_value = 1. - int(sum_sp_value1) / int(sum_sp_value2)
        sp_memory_list2.append(sp_memory2)
        sp_latency_list2.append(sp_latency2)
        sp_flops_list2.append(sp_flops2)
        sparsity_list2.append(sp_value)

    # Plot the metrics
    plot_2metrics(n_qubits_range, sp_memory_list1, sp_memory_list2, cq_memory_list, 'Memory (bytes)', title, 'figure/' + title + "_memory.pdf", logscale=False, show_reduction=True)
    plot_2metrics(n_qubits_range, sp_latency_list1, sp_latency_list2, cq_latency_list, 'Latency (ms)', title, 'figure/' + title + "_latency.pdf", logscale=False, show_reduction=False)
    plot_2metrics(n_qubits_range, sp_flops_list1, sp_flops_list2, cq_flops_list, 'FLOP Count', title, 'figure/' + title + "_flops.pdf", logscale=False, show_reduction=False)
    
    plot_2sparsity(n_qubits_range, sparsity_list1, sparsity_list2, title, 'figure/' + title + "_sparsity.pdf")
    

if __name__ == "__main__":
    # plot_task('vqe', [4, 6])
    plot_task('qft', [2, 3, 4, 5, 6, 7])
    # plot_task('qaoa', [2, 3, 4, 5, 6])
    # plot_2task('pruned_0.5_HWEA3_RYRZ_CZ', 'pruned_0.9_HWEA3_RYRZ_CZ', [50, 100, 150, 200], 'HWEA(N,3,20)')
    # plot_2task('pruned_0.5_HWEA6_RYRZ_CZ', 'pruned_0.9_HWEA6_RYRZ_CZ', [50, 100, 150, 200], 'HWEA(N,6,20)')
    # plot_2task('pruned_0.5_4local', 'pruned_0.9_4local', [50, 100, 150, 200], '4-local')
    plot_task('3local', [50, 100, 150, 200])
    plot_task('aqft_7', [50, 100, 150, 200])