import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_pruned_HWEA_energy_noisefree():
    # Updated data including the VQE result for pruning ratio 0.0
    pruning_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    # vqe_results =  [-7.46838356, -7.62143976, -7.59444344, -7.55981589, -7.59227966, \
    #     -7.07506869, -7.57369852, -7.60075641, -7.59149783, -7.59369878]
    pickle_filename = "experiments/pruned_ansatz_results_noisefree.pkl"
    vqe_results = pickle_load(pickle_filename)
    vqe_results = np.array(vqe_results)
    exact_result = -7.88213996
    
    # Define x positions for boxplots
    x_positions = np.arange(len(pruning_ratios))
    width = 0.3  # Adjust as necessary for visual clarity

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.boxplot(vqe_results, positions=x_positions, widths=width, 
                                 patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.5), showfliers=False,
                                 labels=[f'{pruning_ratio}' for pruning_ratio in pruning_ratios])
    plt.axhline(y=exact_result, color='red', linestyle='--', label="Ground State Energy")

    # Labels and Title
    plt.xlabel("Pruning Ratio", fontsize=16)
    plt.ylabel("Minimal Energy Noisefree", fontsize=16)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title("Energy vs Pruning Ratio")
    plt.legend(fontsize=14)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/pruned_HWEA_energy_noisefree.pdf")
    plt.close()


def plot_pruned_HWEA_energy(filename):
    # Updated data including the VQE result for pruning ratio 0.0
    pruning_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    # vqe_results =  [-7.46838356, -7.62143976, -7.59444344, -7.55981589, -7.59227966, \
    #     -7.07506869, -7.57369852, -7.60075641, -7.59149783, -7.59369878]
    pickle_filename = filename
    vqe_results = pickle_load(pickle_filename)[-10:]
    vqe_results = np.array(vqe_results)
    exact_result = -7.88213996
    
    # Define x positions for boxplots
    x_positions = np.arange(len(pruning_ratios))
    width = 0.3  # Adjust as necessary for visual clarity
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.boxplot(vqe_results, positions=x_positions, widths=width, 
                                 patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.5), showfliers=False,
                                 labels=[f'{pruning_ratio}' for pruning_ratio in pruning_ratios])

    pickle_filename = "experiments/pruned_ansatz_results_noisefree.pkl"
    vqe_results2 = pickle_load(pickle_filename)[-10:]
    vqe_results = np.array(vqe_results2)
    exact_result = -7.88213996
    plt.boxplot(vqe_results, positions=x_positions, widths=width, 
                                 patch_artist=True, boxprops=dict(facecolor='green', alpha=0.5), showfliers=False,
                                 labels=[f'{pruning_ratio}' for pruning_ratio in pruning_ratios])
    plt.axhline(y=exact_result, color='red', linestyle='--', label="Ground State Energy")
    # Labels and Title
    plt.xlabel("Pruning Ratio", fontsize=16)
    plt.ylabel("Minimal Energy with Noise", fontsize=16)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title("Energy vs Pruning Ratio")
    # Create custom legend handles for the two boxplots
    blue_patch = plt.Line2D([0], [0], color='blue', alpha=0.5, lw=4, label='Noisy Results')
    green_patch = plt.Line2D([0], [0], color='green', alpha=0.5, lw=4, label='Noisefree Results')
    plt.legend(handles=[blue_patch, green_patch, plt.Line2D([0], [0], color='red', linestyle='--', label="Ground State Energy")], fontsize=14)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/pruned_HWEA_energy.pdf")
    plt.close()


def plot_two_distributions(exp_name):
    load_filename = f'experiments/{exp_name}.pkl'
    filename = f'figs/{exp_name}.pdf'
    data = pickle_load(load_filename)
    # Sample data from 100 trials of each framework (replace with actual data)
    # qiskit_values = np.random.normal(loc=5, scale=1, size=100)  # Replace with actual qiskit values
    # pgmQC_values = np.random.normal(loc=6, scale=1.2, size=100)  # Replace with actual pgmQC values
    # exact_value = 5.5  # Replace with actual exact value
    
    qiskit_values = np.array(data['qiskit_expval_list']).real
    pgmQC_values = np.array(data['pgmQC_expval_list']).real
    exact_value = data['exact_expval'].real
    # Check if exact_value is a NumPy array and get the scalar value
    if isinstance(exact_value, np.ndarray):
        exact_value = exact_value.item()  # Use .item() to get the scalar value

    # Calculate mean and variance for both sets
    qiskit_mean = np.mean(qiskit_values)
    qiskit_variance = np.var(qiskit_values)

    pgmQC_mean = np.mean(pgmQC_values)
    pgmQC_variance = np.var(pgmQC_values)

    plt.figure(figsize=(8, 8))

    # KDE plot for Qiskit values (blue) with fill=True
    sns.kdeplot(qiskit_values, color="blue", fill=True, alpha=0.5, label='Qiskit')

    # KDE plot for pgmQC values (orange) with fill=True
    sns.kdeplot(pgmQC_values, color="orange", fill=True, alpha=0.5, label='pgmQC')

    # Round the exact value to three decimal places
    exact_value_rounded = round(exact_value, 3)
    plt.axvline(x=exact_value_rounded, color='red', linestyle='--', label=f'Exact Expval = {exact_value_rounded:.3f}', linewidth=2)

    # Plot vertical dotted lines for the means
    plt.axvline(x=qiskit_mean, color='blue', linestyle=':', label=f'Qiskit Mean = {qiskit_mean:.2f}', linewidth=2)
    plt.axvline(x=pgmQC_mean, color='orange', linestyle=':', label=f'pgmQC Mean = {pgmQC_mean:.2f}', linewidth=2)

    # # Place the text for Qiskit mean and variance
    # plt.text(qiskit_mean + 0.1, 0.05, f'Mean={qiskit_mean:.2f}\nVar={qiskit_variance:.2f}', 
    #         color='blue', fontsize=16, verticalalignment='bottom')

    # # Place the text for pgmQC mean and variance in a non-overlapping location
    # plt.text(pgmQC_mean + 0.1, 0.1, f'Mean={pgmQC_mean:.2f}\nVar={pgmQC_variance:.2f}', 
    #         color='orange', fontsize=16, verticalalignment='bottom')

    # Make the axis labels and tick values larger
    plt.xlabel('Reconstructed Expectation Value', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add legend for the exact value and means
    plt.legend(fontsize=14)

    plt.savefig(filename)
    plt.close()

def plot_box_chart_with_classical_shadows(task_name, n_qubits, d, qubit_constrain, total_shots_range, num_trials=1):
    qiskit_expval_all_trials = []
    classical_shadows_all_trials = []
    ourwork_expval_all_trials = []
    exact_value = None
    
    # Loop through the range of total_shots
    for total_shots in total_shots_range:
        load_filename = f'experiments/{task_name}_{n_qubits}_{d}_{qubit_constrain}_{total_shots}_{num_trials}_with_classical_shadows.pkl'
        data = pickle_load(load_filename)  # Assumes you have a function `pickle_load`
        qiskit_expval = np.array(data['qiskit_expval_list']).real
        classical_shadows_expval = np.array(data['classical_shadows_expval_list']).real * (2 ** 8)
        ourwork_expval = np.array(data['pgmQC_expval_list']).real
        exact_value = data['exact_expval'].real
        if isinstance(exact_value, np.ndarray):
            exact_value = exact_value.item()

        qiskit_expval_all_trials.append(qiskit_expval)
        classical_shadows_all_trials.append(classical_shadows_expval)
        ourwork_expval_all_trials.append(ourwork_expval)

    exact_value_rounded = round(exact_value, 3)
    plt.figure(figsize=(5, 8))

    x_positions = np.arange(len(total_shots_range))
    width = 0.3

    plt.boxplot(qiskit_expval_all_trials, positions=x_positions - width, widths=width, 
                patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.5), showfliers=False,
                labels=[f'{shots}' for shots in total_shots_range])
    plt.boxplot(classical_shadows_all_trials, positions=x_positions, widths=width, 
                patch_artist=True, boxprops=dict(facecolor='orange', alpha=0.5), showfliers=False,
                labels=[f'{shots}' for shots in total_shots_range])
    plt.boxplot(ourwork_expval_all_trials, positions=x_positions + width, widths=width, 
                patch_artist=True, boxprops=dict(facecolor='green', alpha=0.5), showfliers=False,
                labels=[f'{shots}' for shots in total_shots_range])

    plt.axhline(y=exact_value_rounded, color='red', linestyle='--', label=f'Exact Value: {exact_value_rounded}')

    # Compute medians
    qiskit_medians = [np.median(trial) for trial in qiskit_expval_all_trials]
    classical_shadows_medians = [np.median(trial) for trial in classical_shadows_all_trials]
    ourwork_medians = [np.median(trial) for trial in ourwork_expval_all_trials]

    for i, x_pos in enumerate(x_positions):
        # Add double arrows for each method
        plt.annotate('', xy=(x_pos - width + 0.25*width, qiskit_medians[i]), xytext=(x_pos - width + 0.25*width, exact_value), 
                     arrowprops=dict(arrowstyle='<->', linestyle=':', color='purple'))
        plt.annotate('', xy=(x_pos + 0.25*width, classical_shadows_medians[i]), xytext=(x_pos + 0.25*width, exact_value), 
                     arrowprops=dict(arrowstyle='<->', linestyle=':', color='purple'))
        plt.annotate('', xy=(x_pos + width + 0.25*width, ourwork_medians[i]), xytext=(x_pos + width + 0.25*width, exact_value), 
                     arrowprops=dict(arrowstyle='<->', linestyle=':', color='purple'))

        y1 = (3 * classical_shadows_medians[i] + 5 * exact_value)/8
        y2 = (ourwork_medians[i] + 3 * exact_value)/4
        # Horizontal reduction arrows
        plt.annotate('', xy=(x_pos - width + 0.25*width, y1), xytext=(x_pos + 0.25*width, y1), 
                     arrowprops=dict(arrowstyle='<-', linestyle='--', color='purple'))
        plt.annotate('', xy=(x_pos - width + 0.25*width, y2), xytext=(x_pos + width + 0.25*width, y2), 
                     arrowprops=dict(arrowstyle='<-', linestyle='--', color='purple'))        # Compute percentage reductions
        reduction_classical = abs(qiskit_medians[i] - classical_shadows_medians[i]) 
        reduction_ourwork = abs(qiskit_medians[i] - ourwork_medians[i])

        # Annotate reductions
        plt.text(x_pos - width / 2 + 0.25*width, y1 + 0.016, 
                 f'-{reduction_classical:.3f}', color='purple', fontsize=14, ha='center', va='center')
        plt.text(x_pos + width / 2 + 0.25*width, y2 + 0.016, 
                 f'-{reduction_ourwork:.3f}', color='purple', fontsize=14, ha='center', va='center')

    plt.ylim(-0.9, 0.9)
    plt.xticks(x_positions, [f'{shots//1000}k' for shots in total_shots_range], fontsize=18)
    plt.xlabel('Total Shots', fontsize=20)
    plt.ylabel('Reconstructed Expval', fontsize=20)

    legend_handles = [plt.Line2D([0], [0], color='blue', lw=4, label='Qiskit'),
                      plt.Line2D([0], [0], color='orange', lw=4, label='Classical Shadows'),
                      plt.Line2D([0], [0], color='green', lw=4, label='Our Work'),
                      plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'Exact Value: {exact_value_rounded}'),
                      plt.Line2D([0], [0], color='purple', lw=2, linestyle='--', label='Error Reduction')]

    plt.legend(handles=legend_handles, fontsize=18, loc='upper center')
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=18)

    filename = f'figs/box_chart_{task_name}_{n_qubits}_{d}_{qubit_constrain}_with_classical_shadows.pdf'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved to {filename}")

def plot_box_chart(task_name, n_qubits, d, qubit_constrain, total_shots_range, num_trials=1):
    qiskit_expval_all_trials = []
    ourwork_expval_all_trials = []
    exact_value = None
    
    # Loop through the range of total_shots
    for total_shots in total_shots_range:
        load_filename = f'experiments/{task_name}_{n_qubits}_{d}_{qubit_constrain}_{total_shots}_{num_trials}.pkl'
        data = pickle_load(load_filename)  # Assumes you have a function `pickle_load`
        qiskit_expval = np.array(data['qiskit_expval_list']).real
        ourwork_expval = np.array(data['pgmQC_expval_list']).real
        exact_value = data['exact_expval'].real
        # Check if exact_value is a NumPy array and get the scalar value
        if isinstance(exact_value, np.ndarray):
            exact_value = exact_value.item()  # Use .item() to get the scalar value

        # Collect all trials for box plot
        qiskit_expval_all_trials.append(qiskit_expval)
        ourwork_expval_all_trials.append(ourwork_expval)

    # Round the exact value to 1e-3
    exact_value_rounded = round(exact_value, 3)

    # Set up the plot with square dimensions
    plt.figure(figsize=(8, 8))  # Larger square-shaped plot (10x10)

    # Create a list of positions for the x-axis, with offsets for Qiskit and our work
    x_positions = np.arange(len(total_shots_range))
    width = 0.3  # Width of each box (Qiskit and our work will be side by side)

    # Box plots for Qiskit and custom work
    boxplot_qiskit = plt.boxplot(qiskit_expval_all_trials, positions=x_positions - width/2, widths=width, 
                                 patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.5), 
                                 labels=[f'{shots}' for shots in total_shots_range])
    boxplot_ourwork = plt.boxplot(ourwork_expval_all_trials, positions=x_positions + width/2, widths=width, 
                                  patch_artist=True, boxprops=dict(facecolor='green', alpha=0.5), 
                                  labels=[f'{shots}' for shots in total_shots_range])

    # Plot the exact value as a horizontal dotted line
    plt.axhline(y=exact_value_rounded, color='red', linestyle='--', label=f'Exact Value: {exact_value_rounded}')
    
    # Set x-axis labels and positions
    plt.xticks(x_positions, [f'{shots//1000}k' for shots in total_shots_range], fontsize=18)  # Larger font for x-axis ticks
    
    plt.ylim(-0.6, 0.5)
    # Increase font sizes for axis labels
    plt.xlabel('Total Shots', fontsize=20)
    plt.ylabel('Reconstructed Expval', fontsize=20)
    
    # Update the legend with correct colors and larger fonts
    legend_handles = [plt.Line2D([0], [0], color='blue', lw=4, label='Qiskit'),
                      plt.Line2D([0], [0], color='green', lw=4, label='Our Work'),
                      plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'Exact Value: {exact_value_rounded}')]

    plt.legend(handles=legend_handles, fontsize=18)

    # Grid and tick parameters
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=18)  # Larger tick labels

    # Save the plot
    filename = f'figs/box_chart_{task_name}_{n_qubits}_{d}_{qubit_constrain}.pdf'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved to {filename}")


def plot_box_chart_qnn(task_name, n_qubits, d, qubit_constrain, total_shots_range, num_trials=1):
    qiskit_acc_all_trials = []
    ourwork_acc_all_trials = []
    exact_value = None
    
    # Loop through the range of total_shots
    for total_shots in total_shots_range:
        len_dataset = 20
        qiskit_acc = np.zeros(num_trials)
        ourwork_acc = np.zeros(num_trials)
        for id in range(len_dataset):
            load_filename = f'experiments/iris_d{d}_shots{total_shots}_dataid{id}_{num_trials}.pkl'
            data = pickle_load(load_filename)  # Assumes you have a function `pickle_load`
            qiskit_acc += np.array(data['qiskit_acc'][-num_trials:])
            ourwork_acc += np.array(data['pgmQC_acc'][-num_trials:])
        exact_value = 1.
        # Check if exact_value is a NumPy array and get the scalar value
        if isinstance(exact_value, np.ndarray):
            exact_value = exact_value.item()  # Use .item() to get the scalar value

        # Collect all trials for box plot
        qiskit_acc_all_trials.append(qiskit_acc / len_dataset)
        ourwork_acc_all_trials.append(ourwork_acc / len_dataset)

    # Round the exact value to 1e-3
    exact_value_rounded = round(exact_value, 3)

    # Set up the plot with square dimensions
    plt.figure(figsize=(8, 5))  # Larger square-shaped plot (10x10)

    # Create a list of positions for the x-axis, with offsets for Qiskit and our work
    x_positions = np.arange(len(total_shots_range))
    width = 0.3  # Width of each box (Qiskit and our work will be side by side)

    # Box plots for Qiskit and custom work
    boxplot_qiskit = plt.boxplot(qiskit_acc_all_trials, positions=x_positions - width/2, widths=width, 
                                 patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.5), 
                                 labels=[f'{shots}' for shots in total_shots_range])
    boxplot_ourwork = plt.boxplot(ourwork_acc_all_trials, positions=x_positions + width/2, widths=width, 
                                  patch_artist=True, boxprops=dict(facecolor='green', alpha=0.5, edgecolor='black', linewidth=2), 
                                  medianprops=dict(color='green', linewidth=2),  # Highlight the median line
                                  labels=[f'{shots}' for shots in total_shots_range])

    # Add scatter points for each data point to make them more noticeable
    for i, trial_data in enumerate(qiskit_acc_all_trials):
        plt.scatter([x_positions[i] - width/2] * len(trial_data), trial_data, color='blue', alpha=0.7, s=50, edgecolor='black')
    # Add scatter points for each data point to make them more noticeable
    for i, trial_data in enumerate(ourwork_acc_all_trials):
        plt.scatter([x_positions[i] + width/2] * len(trial_data), trial_data, color='green', alpha=0.7, s=50, edgecolor='black')

    # Plot the exact value as a horizontal dotted line
    plt.axhline(y=exact_value_rounded, color='red', alpha=0.5, linestyle='--', label=f'Exact Value: {exact_value_rounded}')
    
    # Set x-axis labels and positions
    plt.xticks(x_positions, [f'{shots//1000}k' for shots in total_shots_range], fontsize=22)  # Larger font for x-axis ticks
    plt.ylim(.5, 1.1)
    plt.yticks(np.linspace(0.5, 1, 6), [f"{int(y * 100)}%" for y in np.linspace(0.5, 1, 6)], fontsize=22)
    # Increase font sizes for axis labels
    plt.xlabel('Total Shots', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    
    # Update the legend with correct colors and larger fonts
    legend_handles = [plt.Line2D([0], [0], color='blue', lw=4, label='Qiskit'),
                      plt.Line2D([0], [0], color='green', lw=4, label='Our Work'),
                      plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'Noisefree Accuracy: 100%')]

    plt.legend(handles=legend_handles, fontsize=18)

    # Grid and tick parameters
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=22)  # Larger tick labels

    # Save the plot
    filename = f'figs/box_chart_{task_name}_{n_qubits}_{d}_{qubit_constrain}.pdf'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved to {filename}")



def plot_pruned_HWEA_expressivity_trainability():
    depths = [1, 2, 3, 4]  # Depths of interest
    pruning_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]  # Expanded pruning ratios
    colors = ['blue', 'orange', 'green', 'purple']  # Colors for different depths
    metrics = ['Expressivity', 'Weighted_Expressivity', 'Trainability', 'Overlap']

    for metric in metrics:
        data = {depth: [] for depth in depths}
        medians = {depth: [] for depth in depths}
        std_devs = {depth: [] for depth in depths}  # Store standard deviation for error bars

        for depth in depths:
            filename = f"experiments/pruned_ansatz_results_noisefree_depth{depth}.pkl"
            all_trails_results = pickle_load(filename)

            if metric == 'Expressivity':
                all_trails_data = np.array(all_trails_results['expressivity_lists'])
            elif metric == 'Weighted_Expressivity':
                all_trails_data = np.array(all_trails_results['weighted_expressivity_lists'])
                non_pruned = all_trails_data[:, 0]
                all_trails_data = all_trails_data / non_pruned[:, None]  # Normalize by non-pruned values
            elif metric == 'Overlap':
                all_trails_data = np.array(all_trails_results['expressivity_on_mole_paulistrings_lists'])
            else:
                all_trails_data = np.array(all_trails_results['trainability_lists'])
            
            for ratio in pruning_ratios:
                index = pruning_ratios.index(ratio)  # Get the corresponding index dynamically
                ratio_data = all_trails_data[:, index]
                data[depth].append(ratio_data)
                medians[depth].append(np.median(ratio_data))
                std_devs[depth].append(np.std(ratio_data))  # Compute standard deviation for error bars

        # Create a new figure
        plt.figure(figsize=(7, 5))
        positions = np.array(pruning_ratios)
        width = 0.03  # Width for box plots

        for j, depth in enumerate(depths):
            # Convert data to numpy arrays for consistency
            depth_data = np.array(data[depth], dtype=object).transpose()
            
            # Plot boxplots at each pruning ratio
            plt.boxplot(depth_data, positions=positions+(j-1)*width, widths=width, patch_artist=True, 
                        showfliers=False, boxprops=dict(facecolor=colors[j], alpha=0.3), medianprops=dict(color='black'))

            # Plot median line with shaded error region
            plt.plot(positions+(j-1)*width, medians[depth], marker='o', linestyle='-', color=colors[j], label=f'#Layers: {depth}')
            # plt.fill_between(positions, 
            #                  np.array(medians[depth]) - np.array(std_devs[depth]), 
            #                  np.array(medians[depth]) + np.array(std_devs[depth]), 
            #                  color=colors[j], alpha=0.15)  # Light transparent shading for error bars

        plt.xticks(pruning_ratios, [f"{int(r*100)}%" for r in pruning_ratios], rotation=45)
        plt.xlabel("Pruning Ratio", fontsize=16)
        plt.ylabel(f"{metric} (%)" if metric in ['Expressivity', 'Weighted_Expressivity'] else metric, fontsize=16)

        plt.xlim(-0.05, 1.05)
        if metric in ['Expressivity', 'Weighted_Expressivity', 'Overlap']:
            plt.ylim(0, 1.1)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y * 100)}%'))

        # Add legend (depths now in legend instead of pruning ratios)
        plt.legend(loc='upper right', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'figs/pruned_HWEA_{metric}.pdf')

if __name__ == '__main__':
    # plot_two_distributions('HWEA_5_2_3_100000_100')
    # plot_two_distributions('HWEA_5_3_3_100000_100')
    # plot_two_distributions('HWEA_5_4_3_100000_100')
    # plot_two_distributions('HWEA_RY_CX_5_2_3_100000_100')
    # plot_two_distributions('HWEA_RY_CX_5_3_3_100000_100')
    # plot_two_distributions('HWEA_RY_CX_5_4_3_100000_100')

    # plot_pruned_HWEA_energy_noisefree()
    # plot_pruned_HWEA_energy("experiments/pruned_ansatz_results_noise.pkl")
    # plot_pruned_HWEA_expressivity_trainability()
    for d in [1, 2, 4]:
        plot_box_chart('HWEA_RYRZ_CZ', 8, d, 5, [10000, 20000], 100)
    
    plot_box_chart('3local', 8, 1, 5, [320000, 640000], 100)
    plot_box_chart_qnn('iris', 4, 2, 3, [320000], 10)
    
    plot_box_chart_with_classical_shadows('HWEA_RYRZ_CZ', 8, 2, 5, [10000], 10)
    plot_box_chart_with_classical_shadows('3local', 8, 1, 5, [10000], 10)
    