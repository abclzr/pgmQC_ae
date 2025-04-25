import pdb
import os
import copy
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.colors as mcolors

def plot_magnitude_histogram(values, bins, title, filename):
    # Calculate the magnitudes
    magnitudes = [abs(v) for v in values]

    # Normalize magnitudes to range [0, 1]
    normalized_magnitudes = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes))
    # Define a custom colormap
    colors = ['white', 'blue']
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'custom_blue'
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    
    # Plot the grid using imshow
    plt.imshow(normalized_magnitudes, aspect='equal', cmap=custom_cmap)
    
    # Adding labels and title
    plt.xlabel('Index')
    plt.ylabel('Magnitude of value')
    # plt.title(title)
    
    # Removing y-axis ticks
    plt.yticks([])
    # Adding colorbar
    cbar = plt.colorbar(label='Magnitude')
    cbar.set_label('Magnitude', rotation=270, labelpad=15)  # Rotate the colorbar label
    

    # Saving the plot to a file
    plt.savefig(filename)
    plt.close()

def plot_ps(ps, filename):
    # Plot
    x = np.arange(len(ps))

    plt.bar(x, ps)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Adding labels and title
    plt.xlabel('Index', fontsize=22, fontweight='bold')
    plt.ylabel('Value', fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

def main(task_name, n_qubits, with_noise):
    path = f'../dataset/{task_name}_{n_qubits}/'
    
    arrays_noisefree = pickle.load(open(os.path.join(path, 'arrays_noisefree.pkl'), 'rb'))
    arrays_noise = pickle.load(open(os.path.join(path, 'arrays.pkl'), 'rb'))
    # pdb.set_trace()
    # Example usage
    values = arrays_noise[1].reshape(16, 16)
    bins = 10
    plot_magnitude_histogram(values, bins, '', 'figure/magnitude_histogram_noise.pdf')
    values = arrays_noisefree[1].reshape(16, 16)
    bins = 10
    plot_magnitude_histogram(values, bins, '', 'figure/magnitude_histogram_noisefree.pdf')
    
    values = arrays_noisefree[1].reshape(-1)
    plot_ps(values, f'figure/tensor_value_{task_name}_{n_qubits}.pdf')

if __name__ == '__main__':
    main('qft', 2, False)
    # for n_qubits in [2, 3, 4, 5, 6]:
    #     main('qaoa', n_qubits, False)
    # for n_qubits in [4, 6]:
    #     main('vqe', n_qubits, False)
    # for task_name in ['QFT', 'VQE', 'QAOA']:
    #     for with_noise in [False, True]:
    #         main(task_name, with_noise)