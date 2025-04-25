import numpy as np
import pdb
import os
import cotengra as ctg
import quimb.tensor as qtn
import pickle

from pgmQC.utils import plot_ps

def contract_tensors_and_plot(directory, tensor_list, pickle_filename):
    filename = os.path.join(directory, "description.txt")
    
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
        size_dict[var] = extent

    inputs = []
    inputs_shape = []
    inputs_size = []
    num_tensors = int(lines[3])
    cnt = 3
    for i in range(num_tensors - 1):
        cnt = cnt + 1
        tensor_vars = lines[cnt].split()
        inputs.append(tuple(tensor_vars))
        inputs_shape.append(tuple([size_dict[var] for var in tensor_vars]))
        size = 1
        for var in tensor_vars:
            size = size * size_dict[var]
        inputs_size.append(size)

    output = tuple(lines[cnt + 1].split())

    # Print the parsed data
    print(f"Number of variables: {num_vars}")
    print(f"Variables: {vars}")
    print(f"Extents: {extents}")
    print(f"Number of tensors: {num_tensors}")
    print(f"Inputs: {inputs}")
    print(f"Output: {output}")

    arrays = []
    for tensor, input_shape in zip(tensor_list, inputs_shape):
        arrays.append(tensor.reshape(input_shape))
    
    pickle.dump(arrays, open(os.path.join(directory, pickle_filename), 'wb'))
    
    tn = qtn.TensorNetwork([
        qtn.Tensor(array, inds)
        for array, inds in zip(arrays, inputs)
    ])

    tensor = tn.contract(..., optimize=ctg.HyperOptimizer())
    
    tensor = tensor.data.reshape(-1)
    plot_ps(tensor, os.path.join(directory, f'tensor_cotengra.png'))


def contract_tensor(directory, arrays):
    filename = os.path.join(directory, "description.txt")
    
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
        size_dict[var] = extent

    inputs = []
    inputs_shape = []
    inputs_size = []
    num_tensors = int(lines[3])
    cnt = 3
    for i in range(num_tensors - 1):
        cnt = cnt + 1
        tensor_vars = lines[cnt].split()
        inputs.append(tuple(tensor_vars))
        inputs_shape.append(tuple([size_dict[var] for var in tensor_vars]))
        size = 1
        for var in tensor_vars:
            size = size * size_dict[var]
        inputs_size.append(size)

    output = tuple(lines[cnt + 1].split())
        
    tn = qtn.TensorNetwork([
        qtn.Tensor(array, inds)
        for array, inds in zip(arrays, inputs)
    ])

    tensor = tn.contract(..., optimize=ctg.HyperOptimizer())
    
    tensor = tensor.data.reshape(-1)
    return tensor


def quimb_contraction(arrays, inputs):
    tn = qtn.TensorNetwork([
        qtn.Tensor(array, inds)
        for array, inds in zip(arrays, inputs)
    ])

    tensor = tn.contract(..., optimize=ctg.HyperOptimizer())

    if isinstance(tensor, float):
        return tensor
    return tensor.data.reshape(-1)


def contract_tensor_from_pgmpy(markov_net, uncontracted_nodes):
    # Parse the contents

    inputs = []
    inputs_shape = []
    inputs_size = []
    num_tensors = len(markov_net.factors)
    cnt = 3
    
    arrays = []
    for i in range(num_tensors):
        cnt = cnt + 1
        tensor_vars = markov_net.factors[i].variables
        inputs.append(tuple(tensor_vars))
        inputs_shape.append(tuple([4 for var in tensor_vars]))
        size = 1
        for var in tensor_vars:
            size = size * 4
        inputs_size.append(size)
        arrays.append(markov_net.factors[i].values.reshape(inputs_shape[-1]).real)
    
    tn = qtn.TensorNetwork([
        qtn.Tensor(array, inds)
        for array, inds in zip(arrays, inputs)
    ])

    tensor = tn.contract(..., optimize=ctg.HyperOptimizer())

    if isinstance(tensor, float):
        return tensor
    return tensor
