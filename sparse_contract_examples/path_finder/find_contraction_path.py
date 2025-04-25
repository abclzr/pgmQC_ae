import argparse
import pdb
import os
import cotengra as ctg

def build_path(directory):

    filename = os.path.join(directory, "description.txt")
    output_filename = os.path.join(directory, "contraction_path.txt")

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
    print(f"Number of variables: {num_vars}")
    print(f"Variables: {vars}")
    print(f"Extents: {extents}")
    print(f"Number of tensors: {num_tensors}")
    print(f"Inputs: {inputs}")
    print(f"Output: {output}")
    
    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a binary file.')
    parser.add_argument('directory', type=str, help='The name of the binary file to process')
    args = parser.parse_args()
    directory = args.directory
    build_path(directory)