import argparse
import os
import sys
import subprocess
import argparse
import pdb
import os
import cotengra as ctg

def build_path(directory, start_over=False):

    filename = os.path.join(directory, "description.txt")
    output_filename = os.path.join(directory, "contraction_path.txt")
    if not start_over and os.path.exists(output_filename):
        print(f"{output_filename} already exists, skip finding path")
        return

    # Open the file and read its contents
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Strip whitespace characters from each line
    lines = [line.strip() for line in lines]
    # Parse the contents
    num_vars = int(lines[0])
    vars = lines[1].split()
    extents = list(map(int, lines[2].split()))

    size_dict = {}
    for var, extent in zip(vars, extents):
        size_dict [var] = extent

    inputs = []
    num_tensors = int(lines[3])
    cnt = 3
    for i in range(num_tensors - 1):
        cnt = cnt + 1
        tensor_vars = lines[cnt].split()
        inputs.append(tuple(tensor_vars))

    output = tuple(lines[cnt + 1].split())

    # Print the parsed data
    # print(f"Number of variables: {num_vars}")
    # print(f"Variables: {vars}")
    # print(f"Extents: {extents}")
    # print(f"Number of tensors: {num_tensors}")
    # print(f"Inputs: {inputs}")
    # print(f"Output: {output}")
    
    
    opt = ctg.HyperOptimizer()
    tree = opt.search(inputs, output, size_dict)
    print((tree.contraction_width(), tree.contraction_cost()))
    path = tree.get_path()
    
    tensor_list = list(range(num_tensors - 1))
    parent_id = num_tensors - 1
    with open(output_filename, "w") as file:
        for i, j in path:
            file.write(f"{tensor_list[i]} {tensor_list[j]} {parent_id}\n")
            tensor_list.pop(j)
            tensor_list.pop(i)
            tensor_list.append(parent_id)
            parent_id = parent_id + 1
    
    print(f"Finished. The contractin path has been written to {output_filename}")

# Function to validate directory paths
def validate_directory(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Directory {path} does not exist.")
    return path


def main(dirs, output_dir, start_over=False):
    
    # Main script logic
    print(f"Output Directory: {output_dir}")
    print()

    # Main script logic
    for dir_path in dirs:
        print(f"Directory: {dir_path}")
        print(f"Contents of the directory {dir_path}:")
        print(os.listdir(dir_path))
        print()  # Add a newline for better readability
        build_path(dir_path, start_over)

    output_sptensor_filename = os.path.join(output_dir, "sp_tensors.bin")
    output_sptensor_plaintext_filename = os.path.join(output_dir, "sp_tensors.txt")
    output_sptensor_plaintext_filename2 = os.path.join(output_dir, "sp_result.txt")
    if not start_over and os.path.exists(output_sptensor_filename):
        print(f"{output_sptensor_filename} already exists, skip.")
    else:
        os.system('rm -f ' + output_sptensor_filename)
        os.system('rm -f ' + output_sptensor_plaintext_filename)
        os.system('rm -f ' + output_sptensor_plaintext_filename2)
        for dir_path in dirs:
            os.system(f"./sc -read_mode=non_sparse {dir_path} {output_dir}")
        print(f"{output_sptensor_filename} successfully built.")
    
    build_path(output_dir, start_over)
    os.system(f"./sc -read_mode=sparse {output_dir} {output_dir}")
    print(f"{output_dir} successfully finished contraction.")

def cuquantum_run(dirs, output_dir):
    cmd = f'../cuquantum_examples/build_task {output_dir}'
    for dir in dirs:
        cmd = cmd + f' {dir}'
    os.system(cmd)
    os.system(f'../cuquantum_examples/run_cuquantum_contract {output_dir}')

if __name__ == "__main__":
    start_over = True
    
    # task_names = ['3local']
    task_names = ['aqft_7', '3local']
    for task_name in task_names:
        for n in [50, 100, 150, 200]:
            subpaths = os.listdir(f'../dataset/{task_name}_{n}/')
            subpaths = [subpath for subpath in subpaths if subpath.startswith('subcircuit')]
            dirs = [f'../dataset/{task_name}_{n}/{subpath}' for subpath in subpaths]
            output_dir = f'../dataset/{task_name}_{n}'
            main(dirs, output_dir, start_over)
            cuquantum_run(dirs, output_dir)
    task_names = ['qft']
    for task_name in task_names:
        for n in [2, 3, 4, 5, 6, 7]:
            subpaths = os.listdir(f'../dataset/{task_name}_{n}/')
            subpaths = [subpath for subpath in subpaths if subpath.startswith('subcircuit')]
            dirs = [f'../dataset/{task_name}_{n}/{subpath}' for subpath in subpaths]
            output_dir = f'../dataset/{task_name}_{n}'
            main(dirs, output_dir, start_over)
            cuquantum_run(dirs, output_dir)