import pdb
import os
import time
import pickle
from tqdm import tqdm

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePerth
from qiskit_aer.primitives import EstimatorV2

from qiskit.circuit.library import QFT
from qiskit import transpile
# from pgmpy.inference import VariableElimination
from qiskit.dagcircuit.dagnode import DAGOutNode, DAGInNode, DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
import struct
# from qiskit_experiments.framework import ParallelExperiment
# from qiskit_experiments.library import StateTomography
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeJakartaV2

# from pgmQC.utils.state_transform import density_to_paulistring
from pgmQC.utils import plot_ps, qiskit_wire_cut, qiskit_reconstruction
from pgmQC.model import build_pgm, build_markov_networks, MarkovNetworkBuilder, TensorNetworkBuilder
# from pgmQC.subcircuit_backend.density_matrix import DensityMatrixSimulator
# from pgmQC.subcircuit_backend.noise_model import NoiseModelSimulator
from pgmQC.postprocessing_backend.cotengra_quimb import contract_tensors_and_plot, contract_tensor_from_pgmpy, quimb_contraction
from collections import deque
import networkx as nx
import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_addon_cutting.instructions import CutWire, Move

import networkx as nx
import pgmQC.test_dataset.benchmarks as bm
import math
import argparse
np.random.seed(0)

DEBUG=False

def get_HWEA(n_qubits, d=1, qubit_constrain=4, pruning_ratio=0.):
    cut_position = []
    map_to_physical_qubits = {i: i for i in range(n_qubits)}
    for i in range(1, n_qubits-1):
        if i % (qubit_constrain - 1) == 0:
            cut_position.append(i)
            if i % 2 == 0:
                map_to_physical_qubits[i] = map_to_physical_qubits[i] + 1
            for j in range(i + 1, n_qubits):
                map_to_physical_qubits[j] = map_to_physical_qubits[j] + 1
    circuit = QuantumCircuit(n_qubits)
    circuit_for_qiskit_cut = QuantumCircuit(n_qubits + len(cut_position))
    ob_str = list("I" * (n_qubits + len(cut_position)))
    partition_label = [""] * (n_qubits + len(cut_position))
    cnt = 0
    tmp = 0
    while tmp < n_qubits + len(cut_position):
        for j in range(tmp, min(n_qubits + len(cut_position), tmp + qubit_constrain)):
            partition_label[j] = 'subcircuit_'+ str(cnt)
        cnt = cnt + 1
        tmp = tmp + qubit_constrain
    
    for i in range(d):
        for j in range(n_qubits):
            if DEBUG:
                a = 0
                b = 0
            else:
                a = np.random.uniform(-np.pi, np.pi)
                b = np.random.uniform(-np.pi, np.pi)
            if np.random.rand() > pruning_ratio:
                circuit.ry(a, j)
                circuit_for_qiskit_cut.ry(a, map_to_physical_qubits[j])
            if np.random.rand() > pruning_ratio:
                circuit.rz(b, j)
                circuit_for_qiskit_cut.rz(b, map_to_physical_qubits[j])
        if i % 2 == 0:
            for j in range(0, n_qubits - 1, 2):
                circuit.cz(j, j + 1)
                circuit_for_qiskit_cut.cz(map_to_physical_qubits[j], map_to_physical_qubits[j + 1])
            for j in cut_position:
                # circuit.append(CutWire(), [j])
                if j % 2 == 0:
                    circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j], map_to_physical_qubits[j] - 1])
                    map_to_physical_qubits[j] = map_to_physical_qubits[j] - 1
                else:
                    circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j], map_to_physical_qubits[j] + 1])
                    map_to_physical_qubits[j] = map_to_physical_qubits[j] + 1
            for j in range(1, n_qubits - 1, 2):
                circuit.cz(j, j + 1)
                circuit_for_qiskit_cut.cz(map_to_physical_qubits[j], map_to_physical_qubits[j + 1])
        else:
            for j in range(1, n_qubits - 1, 2):
                circuit.cz(j, j + 1)
                circuit_for_qiskit_cut.cz(map_to_physical_qubits[j], map_to_physical_qubits[j + 1])
            for j in cut_position:
                # circuit.append(CutWire(), [j])
                if j % 2 == 1:
                    circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j], map_to_physical_qubits[j] - 1])
                    map_to_physical_qubits[j] = map_to_physical_qubits[j] - 1
                else:
                    circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j], map_to_physical_qubits[j] + 1])
                    map_to_physical_qubits[j] = map_to_physical_qubits[j] + 1
            for j in range(0, n_qubits - 1, 2):
                circuit.cz(j, j + 1)
                circuit_for_qiskit_cut.cz(map_to_physical_qubits[j], map_to_physical_qubits[j + 1])
    for j in range(n_qubits):
        if DEBUG:
            a = 0
            b = 0
        else:
            a = np.random.uniform(-np.pi, np.pi)
            b = np.random.uniform(-np.pi, np.pi)
            if np.random.rand() > pruning_ratio:
                circuit.ry(a, j)
                circuit_for_qiskit_cut.ry(a, map_to_physical_qubits[j])
            if np.random.rand() > pruning_ratio:
                circuit.rz(b, j)
                circuit_for_qiskit_cut.rz(b, map_to_physical_qubits[j])
        ob_str[map_to_physical_qubits[j]] = "Z"
    observable = SparsePauliOp(["".join(['Z'] * n_qubits)])
    observable_expanded = SparsePauliOp(["".join(reversed(ob_str))])
    
    return circuit, circuit_for_qiskit_cut, observable, observable_expanded, partition_label

def get_HWEA_RYCX(n_qubits, d=1, qubit_constrain=4, pruning_ratio=0.):
    cut_position = []
    map_to_physical_qubits = {i: i for i in range(n_qubits)}
    for i in range(1, n_qubits-1):
        if i % (qubit_constrain - 1) == 0:
            cut_position.append(i)
            if i % 2 == 0:
                map_to_physical_qubits[i] = map_to_physical_qubits[i] + 1
            for j in range(i + 1, n_qubits):
                map_to_physical_qubits[j] = map_to_physical_qubits[j] + 1
    circuit = QuantumCircuit(n_qubits)
    circuit_for_qiskit_cut = QuantumCircuit(n_qubits + len(cut_position))
    ob_str = list("I" * (n_qubits + len(cut_position)))
    partition_label = [""] * (n_qubits + len(cut_position))
    cnt = 0
    tmp = 0
    while tmp < n_qubits + len(cut_position):
        for j in range(tmp, min(n_qubits + len(cut_position), tmp + qubit_constrain)):
            partition_label[j] = 'subcircuit_'+ str(cnt)
        cnt = cnt + 1
        tmp = tmp + qubit_constrain
    
    for i in range(d):
        for j in range(n_qubits):
            if DEBUG:
                a = 0
                b = 0
            else:
                a = np.random.uniform(-np.pi, np.pi)
                b = np.random.uniform(-np.pi, np.pi)
            if np.random.rand() > pruning_ratio:
                circuit.ry(a, j)
                circuit_for_qiskit_cut.ry(a, map_to_physical_qubits[j])
        if i % 2 == 0:
            for j in range(0, n_qubits - 1, 2):
                circuit.cx(j, j + 1)
                circuit_for_qiskit_cut.cx(map_to_physical_qubits[j], map_to_physical_qubits[j + 1])
            for j in cut_position:
                # circuit.append(CutWire(), [j])
                if j % 2 == 0:
                    circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j], map_to_physical_qubits[j] - 1])
                    map_to_physical_qubits[j] = map_to_physical_qubits[j] - 1
                else:
                    circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j], map_to_physical_qubits[j] + 1])
                    map_to_physical_qubits[j] = map_to_physical_qubits[j] + 1
            for j in range(1, n_qubits - 1, 2):
                circuit.cx(j, j + 1)
                circuit_for_qiskit_cut.cx(map_to_physical_qubits[j], map_to_physical_qubits[j + 1])
        else:
            for j in range(1, n_qubits - 1, 2):
                circuit.cx(j, j + 1)
                circuit_for_qiskit_cut.cx(map_to_physical_qubits[j], map_to_physical_qubits[j + 1])
            for j in cut_position:
                # circuit.append(CutWire(), [j])
                if j % 2 == 1:
                    circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j], map_to_physical_qubits[j] - 1])
                    map_to_physical_qubits[j] = map_to_physical_qubits[j] - 1
                else:
                    circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j], map_to_physical_qubits[j] + 1])
                    map_to_physical_qubits[j] = map_to_physical_qubits[j] + 1
            for j in range(0, n_qubits - 1, 2):
                circuit.cx(j, j + 1)
                circuit_for_qiskit_cut.cx(map_to_physical_qubits[j], map_to_physical_qubits[j + 1])
    for j in range(n_qubits):
        if DEBUG:
            a = 0
            b = 0
        else:
            a = np.random.uniform(-np.pi, np.pi)
            b = np.random.uniform(-np.pi, np.pi)
            if np.random.rand() > pruning_ratio:
                circuit.ry(a, j)
                circuit_for_qiskit_cut.ry(a, map_to_physical_qubits[j])
        ob_str[map_to_physical_qubits[j]] = "Z"
    observable = SparsePauliOp(["".join(['Z'] * n_qubits)])
    observable_expanded = SparsePauliOp(["".join(reversed(ob_str))])
    
    return circuit, circuit_for_qiskit_cut, observable, observable_expanded, partition_label

def get_nlocal(n_qubits, d=1, qubit_constrain=4, pruning_ratio=0., n_local=3, flag=None):
    assert qubit_constrain >= 2 * n_local - 1
    cut_start_position_logical = []
    map_to_physical_qubits = {i: i for i in range(n_qubits)}
    i = 0
    while i < n_qubits:
        i = i + qubit_constrain
        if i >= n_qubits:
            break
        i = i - (n_local - 1)
        cut_start_position_logical.append(i)
        for _ in range(i + (n_local - 1), n_qubits):
            map_to_physical_qubits[_] = map_to_physical_qubits[_] + (n_local - 1)
    
    circuit = QuantumCircuit(n_qubits)
    circuit_for_qiskit_cut = QuantumCircuit(n_qubits + len(cut_start_position_logical)*(n_local - 1))
    ob_str = list("I" * (n_qubits + len(cut_start_position_logical)*(n_local - 1)))
    partition_label = [""] * (n_qubits + len(cut_start_position_logical)*(n_local - 1))
    cnt = 0
    tmp = 0
    while tmp < n_qubits + len(cut_start_position_logical)*(n_local - 1):
        for j in range(tmp, min(n_qubits + len(cut_start_position_logical)*(n_local-1), tmp + qubit_constrain)):
            partition_label[j] = 'subcircuit_'+ str(cnt)
        cnt = cnt + 1
        tmp = tmp + qubit_constrain
    
    def do_nlocal_entanglement(circ, start, n_local, flag=None):
        if flag is None or flag == 'RYCZ':
            for j in range(1, n_local):
                circ.cz(start, start + j)
        elif flag == 'RYCX':
            for j in range(1, n_local):
                circ.cx(start, start + j)
    
    for i in range(d):
        for j in range(n_qubits):
            if DEBUG:
                a = 0
                b = 0
            else:
                a = np.random.uniform(-np.pi, np.pi)
                b = np.random.uniform(-np.pi, np.pi)
            if np.random.rand() > pruning_ratio:
                circuit.ry(a, j)
                circuit_for_qiskit_cut.ry(a, map_to_physical_qubits[j])
            if flag is None or flag != 'RYCX':
                if np.random.rand() > pruning_ratio:
                    circuit.rz(b, j)
                    circuit_for_qiskit_cut.rz(b, map_to_physical_qubits[j])
        for j in range(n_qubits - (n_local - 1)):
            do_nlocal_entanglement(circuit, j, n_local, flag)
        j = 0
        while j < n_qubits - (n_local - 1):
            if j in cut_start_position_logical:
                for k in range(n_local - 1):
                    if j + k + n_local - 1 < n_qubits:
                        circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j + k], map_to_physical_qubits[j + k]+n_local-1])
                for k in range(n_local - 1):
                    if j + k + n_local - 1 < n_qubits:
                        do_nlocal_entanglement(circuit_for_qiskit_cut, map_to_physical_qubits[j + k]+n_local-1, n_local, flag)
                for k in range(n_local - 1):
                    if j + k + n_local - 1 < n_qubits:
                        circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j + k]+n_local-1, map_to_physical_qubits[j + k]])
                j = j + n_local-1
            else:
                do_nlocal_entanglement(circuit_for_qiskit_cut, map_to_physical_qubits[j], n_local, flag)
                j = j + 1
    for j in range(n_qubits):
        if DEBUG:
            a = 0
            b = 0
        else:
            a = np.random.uniform(-np.pi, np.pi)
            b = np.random.uniform(-np.pi, np.pi)
            if np.random.rand() > pruning_ratio:
                circuit.ry(a, j)
                circuit_for_qiskit_cut.ry(a, map_to_physical_qubits[j])
            if flag is None or flag != 'RYCX':
                if np.random.rand() > pruning_ratio:
                    circuit.rz(b, j)
                    circuit_for_qiskit_cut.rz(b, map_to_physical_qubits[j])
        ob_str[map_to_physical_qubits[j]] = "Z"
    observable = SparsePauliOp(["".join(['Z'] * n_qubits)])
    observable_expanded = SparsePauliOp(["".join(reversed(ob_str))])
    
    return circuit, circuit_for_qiskit_cut, observable, observable_expanded, partition_label


def adjacent(u, v, dag):
    for edge in dag.edges(u):
        if v in edge:
            return True
    for edge in dag.edges(v):
        if u in edge:
            return True
    return False

def bfs(starts, limits, dag: DAGCircuit):
    visited = starts
    in_dgrees = {}
    for node in dag.nodes():
        in_dgrees[node] = dag._multi_graph.in_degree(node._node_id)

    q = deque()
    for start in starts:
        assert in_dgrees[start] == 0
        q.appendleft(start)
        limits = limits - 1
    
    while limits > 0:
        u = q.popleft()
        for e in dag.edges(u):
            v = e[1]
            if not isinstance(v, DAGOutNode) and adjacent(u, v, dag):
                if v in visited:
                    raise ValueError(f'{v} should not be in the dag, in-degree is still {in_dgrees[v]}')
                in_dgrees[v] = in_dgrees[v] - 1
                if in_dgrees[v] == 0:
                    visited.append(v)
                    q.append(v)
                    limits = limits - 1
                    if limits == 0:
                        break
    
    return {node: node in visited for node in dag.nodes()}

def generate_circuit_by_name(name, n_qubits):
    connected_only = True
    depth = 1

    if name == 'qft':
        print(n_qubits)
        circuit = QuantumCircuit(n_qubits)
        # circuit.h(0)
        # circuit.cx(0, 1)
        circuit = QuantumCircuit(n_qubits)
        circuit = QFT(num_qubits=n_qubits, approximation_degree=0, do_swaps=False, inverse=False, insert_barriers=False, name='qft')
        circuit = transpile(circuit, basis_gates=['cx', 'u3'])
        return circuit
    elif name == 'ghz':
        circuit = bm.ghz(num_qubits=n_qubits)
        circuit = transpile(circuit, basis_gates=['cx', 'u3'])
        return circuit
    elif name == 'wstate':
        circuit = bm.w_state(n_qubits)
        circuit = transpile(circuit, basis_gates=['cx', 'u3', 'ry', 'x', 'cz'])
        return circuit
    elif name == 'supremacy':
        grid_length = int(math.sqrt(n_qubits))
        assert (
            grid_length**2 == n_qubits
        ), "Supremacy is defined for n*n square circuits"
        circuit = bm.gen_supremacy(
            grid_length, grid_length, depth * 8, order="random", regname="q"
        )
        circuit = transpile(circuit, basis_gates=['cx', 'cz', 'u3', 't', 'x', 'y'])
        return circuit
    elif name == 'sycamore':
        grid_length = int(math.sqrt(n_qubits))
        assert (
            grid_length**2 == n_qubits
        ), "Sycamore is defined for n*n square circuits"
        circuit = bm.gen_sycamore(grid_length, grid_length, depth * 8, regname="q")
        circuit = transpile(circuit, basis_gates=['cx', 'cz', 'u3', 't', 'x', 'y'])
        return circuit
    elif name == 'erdos':
        random_density = np.random.uniform(0, 1)
        graph = nx.generators.random_graphs.erdos_renyi_graph(
            n_qubits, random_density
        )
        if connected_only:
            density_delta = 0.001
            lo = 0
            hi = 1
            while lo < hi:
                mid = (lo + hi) / 2
                graph = nx.generators.random_graphs.erdos_renyi_graph(
                    n_qubits, mid
                )
                if nx.number_connected_components(graph) == 1:
                    hi = mid - density_delta
                else:
                    lo = mid + density_delta
        circuit = bm.construct_qaoa_plus(
            P=depth,
            G=graph,
            params=[np.random.uniform(-np.pi, np.pi) for _ in range(2 * depth)],
            reg_name="q",
        )

        circuit = transpile(circuit, basis_gates=['cx', 'u3', 'rx', 'rz'])
        return circuit
    elif name == 'HWEA':
        return get_HWEA(n_qubits)
        

def generate_uncontracted_tensors(markov_net, uncontracted_nodes, path):
    filename = path + 'description.txt'
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    var_list = []
    for factor in markov_net.factors:
        var_list = var_list + factor.variables
    var_list = list(set(var_list))
    num_variables = len(var_list)
    # pdb.set_trace()
    with open(filename, "w") as file:
        file.write(str(num_variables) + '\n')
        for var in var_list:
            file.write(var + " ")
        file.write('\n')
        for var in var_list:
            file.write('4' + " ")
        file.write('\n')
        
        file.write(str(len(markov_net.factors) + 1) + '\n')
        for factor in markov_net.factors:
            for var in factor.variables:
                file.write(var + " ")
            file.write('\n')
        for var in uncontracted_nodes:
            file.write(var + " ")
        file.write('\n')
    
    binary_file = path + 'tensors.bin'
    with open(binary_file, "wb") as file:
        # Iterate through the float_list
        for factor in markov_net.factors:
            for fp in factor.values.real.reshape(-1):
                # Convert float to binary data using struct.pack
                float_bytes = struct.pack('f', fp)
                # Write the binary data to the file
                file.write(float_bytes)

def pickle_store(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def get_backend():
    # return AerSimulator(method='statevector')
    # return FakeManilaV2()
    return FakeJakartaV2()

def get_aqft(n_qubits, n_local, qubit_constrain):
    circuit = QuantumCircuit(n_qubits)
    n_physical_qubits = int(np.ceil((n_qubits - n_local + 1) / (qubit_constrain - n_local + 1)) - 1) * (n_local - 1) + n_qubits
    circuit_for_qiskit_cut = QuantumCircuit(n_physical_qubits)
    observable_expanded_str = list("I" * n_physical_qubits)
    partition_label = [""] * n_physical_qubits
    subcircuit_start_logical = 0
    subcircuit_start_physical = 0
    subcircuit_cnt = 0
    for i in range(n_qubits):
        circuit.h(i)
        circuit_for_qiskit_cut.h(subcircuit_start_physical + i - subcircuit_start_logical)
        observable_expanded_str[subcircuit_start_physical + i - subcircuit_start_logical] = "Z"
        partition_label[subcircuit_start_physical + i - subcircuit_start_logical] = 'subcircuit_' + str(subcircuit_cnt)
        for j in range(i+1, min(i+n_local, n_qubits)):
            circuit.cp(2*np.pi/(2 ** (j-i+1)), j, i)
            circuit_for_qiskit_cut.cp(2*np.pi/(2 ** (j-i+1)), subcircuit_start_physical + j - subcircuit_start_logical, subcircuit_start_physical + i - subcircuit_start_logical)
        if min(i + n_local, n_qubits-1) >= subcircuit_start_logical + qubit_constrain:
            for k in range(subcircuit_start_physical + qubit_constrain - 1 - n_local + 2, subcircuit_start_physical + qubit_constrain):
                circuit_for_qiskit_cut.append(Move(), [k, k + n_local - 1])
                partition_label[k] = 'subcircuit_' + str(subcircuit_cnt)
            subcircuit_start_physical += qubit_constrain
            subcircuit_start_logical += qubit_constrain - n_local + 1
            subcircuit_cnt += 1
    
    observable = SparsePauliOp(["".join(['Z'] * n_qubits)])
    observable_expanded = SparsePauliOp(["".join(reversed(observable_expanded_str))])
    
    return circuit, circuit_for_qiskit_cut, observable, observable_expanded, partition_label

def build_qnn(n_qubits, d, qubit_constrain, X, weight):
    circ = QuantumCircuit(n_qubits)
    circ_for_qiskit_cut = QuantumCircuit(n_qubits + 1)
    l_to_p = {0 : 0, 1 : 1, 2 : 2, 3 : 4}
    for i in range(n_qubits):
        circ.h(i)
        circ.rz(X[i], i)
        circ_for_qiskit_cut.h(l_to_p[i])
        circ_for_qiskit_cut.rz(X[i], l_to_p[i])
    for i in range(n_qubits - 1):
        circ.cx(i, i + 1)
        circ.rz(X[n_qubits + i], i + 1)
        circ.cx(i, i + 1)
        if i + 1 <= 2:
            circ_for_qiskit_cut.cx(i, i + 1)
            circ_for_qiskit_cut.rz(X[n_qubits + i], i + 1)
            circ_for_qiskit_cut.cx(i, i + 1)
        else:
            circ_for_qiskit_cut.append(Move(), [2, 3])
            circ_for_qiskit_cut.cx(3, 4)
            circ_for_qiskit_cut.rz(X[n_qubits + i], 4)
            circ_for_qiskit_cut.cx(3, 4)
            circ_for_qiskit_cut.append(Move(), [3, 2])
    cnt = 0
    for _ in range(d):
        for i in range(n_qubits):
            circ.ry(weight[cnt], i)
            circ_for_qiskit_cut.ry(weight[cnt], l_to_p[i])
            cnt += 1
        for i in range(n_qubits - 1):
            circ.cx(i, i + 1)
            if i + 1 <= 2:
                circ_for_qiskit_cut.cx(i, i + 1)
            else:
                circ_for_qiskit_cut.append(Move(), [2, 3])
                circ_for_qiskit_cut.cx(3, 4)
                if _ != d - 1:
                    circ_for_qiskit_cut.append(Move(), [3, 2])
    
    l_to_p[2] = 3
    for i in range(n_qubits):
        circ.ry(weight[cnt], i)
        circ_for_qiskit_cut.ry(weight[cnt], l_to_p[i])
        cnt += 1
    
    observable = {"predict_0": SparsePauliOp(["IZII"]), 
                 "predict_1": SparsePauliOp(["ZIII"])}
    observable_expanded = {"predict_0": SparsePauliOp(["IZIII"]),
                            "predict_1": SparsePauliOp(["ZIIII"])}
    partion_label = ["subcircuit_0"] * 3 + ["subcircuit_1"] * 2
    return circ, circ_for_qiskit_cut, observable, observable_expanded, partion_label

def build_qnn_d2(n_qubits, d, qubit_constrain, X, weight):
    circ = QuantumCircuit(n_qubits)
    circ_for_qiskit_cut = QuantumCircuit(n_qubits + 2)
    l_to_p = {0 : 0, 1 : 1, 2 : 2, 3 : 5}
    for i in range(n_qubits):
        circ.h(i)
        circ.rz(X[i], i)
        circ_for_qiskit_cut.h(l_to_p[i])
        circ_for_qiskit_cut.rz(X[i], l_to_p[i])
    for i in range(n_qubits - 1):
        circ.cx(i, i + 1)
        circ.rz(X[n_qubits + i], i + 1)
        circ.cx(i, i + 1)
        if i + 1 <= 2:
            circ_for_qiskit_cut.cx(i, i + 1)
            circ_for_qiskit_cut.rz(X[n_qubits + i], i + 1)
            circ_for_qiskit_cut.cx(i, i + 1)
        else:
            circ_for_qiskit_cut.append(Move(), [2, 4])
            circ_for_qiskit_cut.cx(4, 5)
            circ_for_qiskit_cut.rz(X[n_qubits + i], 5)
            circ_for_qiskit_cut.cx(4, 5)
            circ_for_qiskit_cut.append(Move(), [4, 2])
    cnt = 0
    for _ in range(d):
        for i in range(n_qubits):
            circ.ry(weight[cnt], i)
            circ_for_qiskit_cut.ry(weight[cnt], l_to_p[i])
            cnt += 1
        for i in range(n_qubits - 1):
            circ.cx(i, i + 1)
            if _ == 0:
                if i + 1 <= 2:
                    circ_for_qiskit_cut.cx(i, i + 1)
                else:
                    circ_for_qiskit_cut.append(Move(), [2, 4])
                    circ_for_qiskit_cut.cx(4, 5)
                    l_to_p[2] = 4
                    break
            elif _ == 1:
                if i + 1 <= 1:
                    circ_for_qiskit_cut.cx(i, i + 1)
                else:
                    circ.cx(2, 3)
                    circ_for_qiskit_cut.append(Move(), [1, 3])
                    circ_for_qiskit_cut.cx(3, 4)
                    circ_for_qiskit_cut.cx(4, 5)
                    l_to_p[1] = 3
                    break
    
    for i in range(n_qubits):
        circ.ry(weight[cnt], i)
        circ_for_qiskit_cut.ry(weight[cnt], l_to_p[i])
        cnt += 1
    
    observable = {"predict_0": SparsePauliOp(["IZII"]), 
                 "predict_1": SparsePauliOp(["ZIII"])}
    observable_expanded = {"predict_0": SparsePauliOp(["IZIIII"]),
                            "predict_1": SparsePauliOp(["ZIIIII"])}
    partion_label = ["subcircuit_0"] * 3 + ["subcircuit_1"] * 3
    return circ, circ_for_qiskit_cut, observable, observable_expanded, partion_label

def sparsity(arr, tol=1e-6):
    close_to_zero = np.abs(arr) < tol
    # Calculate the percentage
    sp = 1 - np.sum(close_to_zero) / arr.size
    return sp

def evaluate_one_data(circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label, total_shots, with_classical_shadows, num_trials):
    qiskit_result = {}
    classical_shadow_result = {}
    pgmQC_result = {}
    for y_label in ["predict_0", "predict_1"]:
        print("Evaluating %s" % y_label)
        subcircuits, subobservables, bases = qiskit_wire_cut(circuit, observable_expanded[y_label], partition_label, 4)

        # prepare for pgmQC subcircuits
        for i in range(len(subcircuits)):
            subcircuits[f'subcircuit_{i}'].draw("mpl", scale=0.8, filename=f'subcircuit_{i}.png')

        tensor_network_builders = {
            label: TensorNetworkBuilder(circuit_to_dag(subcircuit), label)
            for label, subcircuit in subcircuits.items()
        }
        arrays = []
        inds = []
        pgmQC_total_experiments = 0
        classical_shadows_total_inputs = 0
        remained_experiments_ratio = []
        for label, tensor_network_builder in tensor_network_builders.items():
            tensor_net, uncontracted_nodes = tensor_network_builder.build(observable=observable_expanded[y_label].paulis[0])
            print(tensor_network_builder.closed_dag_out_nodes_Pauli)
            contracted_tensor = contract_tensor_from_pgmpy(tensor_net, uncontracted_nodes)
            arrays.append(contracted_tensor.data)
            inds.append(contracted_tensor.inds)
            true_false_tensor = contract_tensor_from_pgmpy(tensor_network_builder.true_false_network, uncontracted_nodes)
            print(sparsity(true_false_tensor.data))
            num_subcircuit_experiments = tensor_network_builder.num_experiments(true_false_tensor.data, true_false_tensor.inds)
            pgmQC_total_experiments += num_subcircuit_experiments
            remained_experiments_ratio.append(num_subcircuit_experiments\
                / ((4 ** len(tensor_network_builder.input_dag_nodes)) * (3 ** len(tensor_network_builder.output_dag_nodes))))
            classical_shadows_total_inputs += 4 ** len(tensor_network_builder.input_dag_nodes)
        print(f'exact expval: {quimb_contraction(arrays, inds)}')
        
        print('======================qiskit addon cutting======================')
        print(f"Sampling overhead: {np.prod([basis.overhead for basis in bases])}")

        backend = get_backend()
        expval_list, metrics = qiskit_reconstruction(backend, subcircuits, subobservables, observable[y_label], total_shots, num_trials)
        qiskit_result[y_label] = expval_list
        print(f'Qiskit expval: {expval_list}')
        if with_classical_shadows:
            print('======================classical shadows======================')
            num_classical_shadows_per_input_state = total_shots // classical_shadows_total_inputs
            print(f'Total input states: {classical_shadows_total_inputs}')
            print(f'Total shots: {total_shots}')
            print(f'Classical shadows per input state: {num_classical_shadows_per_input_state}')
            backend = get_backend()
            classical_shadows_expval_list = []
            classical_shadows_execution_time_ms_list = []

            for i in tqdm(range(num_trials)):
                sampled_tensors = []
                tensor_varnames = []
                for label, tensor_network_builder in tensor_network_builders.items():
                    sampled_tensor, varname = tensor_network_builder.evaluate_by_classical_shadows(backend, num_classical_shadows_per_input_state)
                    sampled_tensors.append(sampled_tensor)
                    tensor_varnames.append(tuple(varname))
                start_time = time.time()
                expval = quimb_contraction(sampled_tensors, tensor_varnames)
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000
                classical_shadows_execution_time_ms_list.append(execution_time_ms)
                classical_shadows_expval_list.append(expval)
            
            classical_shadow_result[y_label] = classical_shadows_expval_list
        
        print('======================pgmQC======================')
        shots_per_experiment = total_shots // pgmQC_total_experiments
        print(f'Total experiments: {pgmQC_total_experiments}')
        print(f'Total shots: {total_shots}')
        print(f'Shots per experiment: {shots_per_experiment}')
        backend = get_backend()
        pgmQC_expval_list = []
        pgmQC_execution_time_ms_list = []
        for i in tqdm(range(num_trials)):
            sampled_tensors = []
            tensor_varnames = []
            for label, tensor_network_builder in tensor_network_builders.items():
                sampled_tensor, varname = tensor_network_builder.evaluate_by_sampling(backend, shots=shots_per_experiment)
                sampled_tensors.append(sampled_tensor)
                tensor_varnames.append(tuple(varname))
            start_time = time.time()
            expval = quimb_contraction(sampled_tensors, tensor_varnames)
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            pgmQC_execution_time_ms_list.append(execution_time_ms)
            pgmQC_expval_list.append(expval)
        
        pgmQC_result[y_label] = pgmQC_expval_list
        print(f'pgmQC expval: {pgmQC_expval_list}')
    
    qiskit_predict = []
    classical_shadow_predict = []
    pgmQC_predict = []
    for i in range(num_trials):
        if qiskit_result["predict_0"][i] > qiskit_result["predict_1"][i]:
            qiskit_predict.append(0)
        else:
            qiskit_predict.append(1)
        if with_classical_shadows:
            if classical_shadow_result["predict_0"][i] > classical_shadow_result["predict_1"][i]:
                classical_shadow_predict.append(0)
            else:
                classical_shadow_predict.append(1)
        if pgmQC_result["predict_0"][i] > pgmQC_result["predict_1"][i]:
            pgmQC_predict.append(0)
        else:
            pgmQC_predict.append(1)
    return qiskit_predict, classical_shadow_predict, pgmQC_predict

def test_qnn(n_qubits, d, qubit_constrain, total_shots_range, num_trials, with_classical_shadows, dataset_start, dataset_end):
    with open(f'iris_model_d{d}.pkl', 'rb') as f:
        model = pickle.load(f)
    X_test = model['X_test']
    y_test = model['y_test']
    predicts = model['predict']
    weight = model['weight']
    final_results = []
    for total_shots in total_shots_range:
        print(f"Total shots: {total_shots}")
        results = {'total_shots': total_shots, 'qiskit_acc': [], 'classical_shadow_acc': [], 'pgmQC_acc': []}
        qiskit_predicts = []
        classical_shadow_predicts = []
        pgmQC_predicts = []
        for __ in range(dataset_start, dataset_end):
            X = X_test[__]
            y = y_test[__]
            predict = predicts[__]
            print(f"X: {X}, y: {y}, predict: {predict}")
            if d == 1:
                circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = build_qnn(n_qubits, d, qubit_constrain, X, weight)
            elif d == 2:
                circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = build_qnn_d2(n_qubits, d, qubit_constrain, X, weight)
            circuit_for_pgmQC.draw("mpl", scale=0.8, filename='original_circuit.png')
            qiskit_predict, classical_shadow_predict, pgmQC_predict = \
                evaluate_one_data(circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label, total_shots, with_classical_shadows, num_trials)
            qiskit_predicts.append(qiskit_predict)
            if with_classical_shadows:
                classical_shadow_predicts.append(classical_shadow_predict)
            pgmQC_predicts.append(pgmQC_predict)
            print(f'label: {y}')
            print(f'Noisefree predict: {predict}')
            print(f'Qiskit predict: {qiskit_predict}')
            if with_classical_shadows:
                print(f'Classical shadow predict: {classical_shadow_predict}')
            print(f'PGMQC predict: {pgmQC_predict}')
        
            print(f'Qiskit predicts: {qiskit_predicts}')
            if with_classical_shadows:
                print(f'Classical shadow predicts: {classical_shadow_predicts}')
            print(f'PGMQC predicts: {pgmQC_predicts}')
            for trial_id in range(num_trials):
                qiskit_win = 0
                classical_shadow_win = 0
                pgmQC_win = 0
                if qiskit_predict[trial_id] == y:
                    qiskit_win += 1
                if with_classical_shadows:
                    if classical_shadow_predict[trial_id] == y:
                        classical_shadow_win += 1
                if pgmQC_predict[trial_id] == y:
                    pgmQC_win += 1
                print(f"Trial {trial_id}: Qiskit win: {qiskit_win}, Classical shadow win: {classical_shadow_win}, PGMQC win: {pgmQC_win}")
                results['qiskit_acc'].append(qiskit_win)
                if with_classical_shadows:
                    results['classical_shadow_acc'].append(classical_shadow_win)
                results['pgmQC_acc'].append(pgmQC_win)
            final_results.append(results)
            pickle_store(results, f'experiments/iris_d{d}_shots{total_shots}_dataid{__}_{num_trials}.pkl')
    return final_results

def main(task_name, n_qubits, d, qubit_constrain, total_shots_range, num_trials=1, with_classical_shadows=False):    
    print(f"Processing {task_name} with {n_qubits} qubits, depth {d}, qubit constrain {qubit_constrain}, total shots {total_shots_range}, num trials {num_trials}, with classical shadows {with_classical_shadows}")
    # circuit = generate_circuit_by_name(task_name, n_qubits)
    while True:
        if task_name == 'aqft':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_aqft(n_qubits, d, qubit_constrain)
            print(f"AQFT Precision: π/{(2 ** (d-1))}")
            break
        elif task_name == 'HWEA_RYRZ_CZ':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_HWEA(n_qubits, d, qubit_constrain)
        elif task_name == 'HWEA_RY_CX':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = bm.get_HWEA_Michele_Grossi(n_qubits, d, qubit_constrain)
        elif task_name == '2local':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 2)
        elif task_name == '3local':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 3)
        elif task_name == '2local_RYCX':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 2, 'RYCX')
        elif task_name == '3local_RYCX':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 3, 'RYCX')
        elif task_name == '2local_RYCZ':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 2, 'RYCZ')
        elif task_name == '3local_RYCZ':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 3, 'RYCZ')
        elif task_name == '4local':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 4)
        elif task_name == '4local_RYCX':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 4, 'RYCX')
        elif task_name == '4local_RYCZ':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 4, 'RYCZ')
        else:
            raise ValueError(f'Unknown task name: {task_name}')
        
        estimator = EstimatorV2()
        exact_expval = estimator.run([(circuit_for_pgmQC, observable)]).result()[0].data.evs
        if -0.4 < exact_expval and exact_expval < -0.3:
            print('Satisfy the requirement (-0.4 < expval < -0.3, to make the plot looks clearer and consistent)')
            break
        else:
            print(f"\rExact expectation value: {np.round(exact_expval, 8)}, retrying...", end="")
    
    circuit_for_pgmQC.draw("mpl", scale=0.8, filename='original_circuit.png')
    circuit.draw("mpl", scale=0.8, filename='circuit.png')
    # testing circuit_for_pgmQC
    # tnb = TensorNetworkBuilder(circuit_to_dag(circuit_for_pgmQC), 'o')
    # tensor_net, uncontracted_nodes = tnb.build(observable=observable.paulis[0], show_density_matrix=True)
    # contracted_tensor = contract_tensor_from_pgmpy(tensor_net, uncontracted_nodes)
    # print(contracted_tensor)
    # pdb.set_trace()
    # prepare for qiskit subcircuits
    subcircuits, subobservables, bases = qiskit_wire_cut(circuit, observable_expanded, partition_label, 4)

    # prepare for pgmQC subcircuits
    for i in range(len(subcircuits)):
        subcircuits[f'subcircuit_{i}'].draw("mpl", scale=0.8, filename=f'subcircuit_{i}.png')

    tensor_network_builders = {
        label: TensorNetworkBuilder(circuit_to_dag(subcircuit), label)
        for label, subcircuit in subcircuits.items()
    }
    arrays = []
    inds = []
    pgmQC_total_experiments = 0
    classical_shadows_total_inputs = 0
    remained_experiments_ratio = []
    for label, tensor_network_builder in tensor_network_builders.items():
        tensor_net, uncontracted_nodes = tensor_network_builder.build()
        contracted_tensor = contract_tensor_from_pgmpy(tensor_net, uncontracted_nodes)
        arrays.append(contracted_tensor.data)
        inds.append(contracted_tensor.inds)
        true_false_tensor = contract_tensor_from_pgmpy(tensor_network_builder.true_false_network, uncontracted_nodes)
        print(sparsity(true_false_tensor.data))
        num_subcircuit_experiments = tensor_network_builder.num_experiments(true_false_tensor.data, inds[-1])
        pgmQC_total_experiments += num_subcircuit_experiments
        remained_experiments_ratio.append(num_subcircuit_experiments\
            / ((4 ** len(tensor_network_builder.input_dag_nodes)) * (3 ** len(tensor_network_builder.output_dag_nodes))))
        classical_shadows_total_inputs += 4 ** len(tensor_network_builder.input_dag_nodes)
    print(f'exact expval: {quimb_contraction(arrays, inds)}')
    
    for total_shots in total_shots_range:
        if with_classical_shadows:
            filename = f'experiments/{task_name}_{n_qubits}_{d}_{qubit_constrain}_{total_shots}_{num_trials}_with_classical_shadows.pkl'
        else:
            filename = f'experiments/{task_name}_{n_qubits}_{d}_{qubit_constrain}_{total_shots}_{num_trials}.pkl'
        
        if os.path.exists(filename):
            print(f"File {filename} already exists. Skipping this iteration.")
            continue
        
        print('======================qiskit addon cutting======================')
        print(f"Sampling overhead: {np.prod([basis.overhead for basis in bases])}")

        backend = get_backend()
        expval_list, metrics = qiskit_reconstruction(backend, subcircuits, subobservables, observable, total_shots, num_trials=num_trials)
        
        print('Mean of estimation: ', np.mean(expval_list))
        print('Variance of estimation: ', np.var(expval_list))
        result = {'qiskit_expval_list': expval_list, 'exact_expval': exact_expval}
        result.update(metrics)
        # pickle_store(result, filename)
        # print(f"Error in estimation: {np.real(np.round(reconstructed_expval-exact_expval, 8))}")
        # print(
        #     f"Relative error in estimation: {np.real(np.round((reconstructed_expval-exact_expval) / exact_expval, 8))}"
        # )
        
        # backend = AerSimulator(method='statevector')
        if with_classical_shadows:
            print('======================classical shadows======================')
            num_classical_shadows_per_input_state = total_shots // classical_shadows_total_inputs
            print(f'Total input states: {classical_shadows_total_inputs}')
            print(f'Total shots: {total_shots}')
            print(f'Classical shadows per input state: {num_classical_shadows_per_input_state}')
            backend = get_backend()
            classical_shadows_expval_list = []
            classical_shadows_execution_time_ms_list = []

            for i in tqdm(range(num_trials)):
                sampled_tensors = []
                tensor_varnames = []
                for label, tensor_network_builder in tensor_network_builders.items():
                    sampled_tensor, varname = tensor_network_builder.evaluate_by_classical_shadows(backend, num_classical_shadows_per_input_state)
                    sampled_tensors.append(sampled_tensor)
                    tensor_varnames.append(tuple(varname))
                start_time = time.time()
                expval = quimb_contraction(sampled_tensors, tensor_varnames)
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000
                classical_shadows_execution_time_ms_list.append(execution_time_ms)
                classical_shadows_expval_list.append(expval)
            
            print(f'Mean of estimation: ', np.mean(classical_shadows_expval_list))
            print(f'Variance of estimation: ', np.var(classical_shadows_expval_list))
            result['classical_shadows_expval_list'] = classical_shadows_expval_list
            result.update({
                'classical_shadows_total_input_states': classical_shadows_total_inputs,
                'classical_shadows_shots_per_input_state': num_classical_shadows_per_input_state,
                'classical_shadows_total_shots': total_shots,
                'classical_shadows_execution_time_ms_list': classical_shadows_total_inputs,
            })
            pickle_store(result, filename)
        
        print('======================pgmQC======================')
        shots_per_experiment = total_shots // pgmQC_total_experiments
        print(f'Total experiments: {pgmQC_total_experiments}')
        print(f'Total shots: {total_shots}')
        print(f'Shots per experiment: {shots_per_experiment}')
        backend = get_backend()
        pgmQC_expval_list = []
        pgmQC_execution_time_ms_list = []
        for i in tqdm(range(num_trials)):
            sampled_tensors = []
            tensor_varnames = []
            for label, tensor_network_builder in tensor_network_builders.items():
                sampled_tensor, varname = tensor_network_builder.evaluate_by_sampling(backend, shots=shots_per_experiment)
                sampled_tensors.append(sampled_tensor)
                tensor_varnames.append(tuple(varname))
            start_time = time.time()
            expval = quimb_contraction(sampled_tensors, tensor_varnames)
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            pgmQC_execution_time_ms_list.append(execution_time_ms)
            pgmQC_expval_list.append(expval)
        
        print(f'Mean of estimation: ', np.mean(pgmQC_expval_list))
        print(f'Variance of estimation: ', np.var(pgmQC_expval_list))
        result['pgmQC_expval_list'] = pgmQC_expval_list
        result.update({
            'pgmQC_total_experiments': pgmQC_total_experiments,
            'pgmQC_shots_per_experiment': shots_per_experiment,
            'pgmQC_total_shots': total_shots,
            'pgmQC_execution_time_ms_list': pgmQC_execution_time_ms_list,
            'pgmQC_remained_experiments_ratio': remained_experiments_ratio
        })
        pickle_store(result, filename)
        # dag = circuit_to_dag(circuit_for_pgmQC)
        # markov_net_builder = MarkovNetworkBuilder(dag, noise_inject=False)
        # markov_net, uncontracted_nodes = markov_net_builder.build()
        # print(contract_tensor_from_pgmpy(markov_net, uncontracted_nodes))
    
if __name__ == '__main__':
    # main('3local', 8, 1, 5, [320000, 640000, 1280000], 100)
    
    # main('HWEA_RYRZ_CZ', 8, 1, 5, [10000, 20000, 40000, 80000, 160000], 100)
    # main('2local', 8, 1, 5, [10000], 100, True)
    # main('3local', 8, 1, 5, [10000], 100, True)
    # main('4local', 10, 1, 7, [10000], 100, False)
    # main('2local_RYCX', 8, 1, 5, [10000], 100, True)
    # main('2local_RYCX', 8, 2, 5, [10000], 100, True)
    # main('2local_RYCX', 8, 3, 5, [10000], 100, True)
    # main('3local_RYCX', 8, 1, 5, [10000], 100, True)
    # main('4local_RYCX', 20, 1, 13, [10000], 100, False)
    
    # main('2local_RYCZ', 8, 1, 5, [10000], 100, True)
    # main('2local_RYCZ', 8, 2, 5, [10000], 100, True)
    # main('2local_RYCZ', 8, 3, 5, [10000], 100, True)
    
    # main('HWEA_RYRZ_CZ', 8, 2, 5, [10000, 20000, 40000, 80000, 160000], 100)
    # main('HWEA_RYRZ_CZ', 8, 4, 5, [10000, 20000, 40000, 80000, 160000], 100)
    # main('HWEA_RYRZ_CZ', 8, 4, 5, [320000, 640000, 1280000], 100)
    # main('aqft', 8, 3, 5, [10000], 100, True)
    parser = argparse.ArgumentParser(description="Process dataset range for QNN.")
    parser.add_argument("--task_name", type=str, default="QNN", help="Task name.")
    parser.add_argument("--n_qubits", type=int, default=8, help="Number of qubits.")
    parser.add_argument("--d", type=int, default=1, help="Depth of the circuit.")
    parser.add_argument("--qubit_constrain", type=int, default=5, help="Qubit constraint.")
    parser.add_argument("--dataset_range", type=str, default="0-20", help="Range of dataset to process, e.g., '0-10'.")
    parser.add_argument("--total_shots", type=int, default=320000, help="Total shots for the experiment.")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of trials for the experiment.")
    parser.add_argument("--with_classical_shadows", action="store_true", help="Use classical shadows.")
    args = parser.parse_args()
    if args.task_name == "QNN":
        # Parse the dataset range
        dataset_start, dataset_end = map(int, args.dataset_range.split('-'))
        total_shots = args.total_shots
        print(f"Processing QNN dataset range: {dataset_start} to {dataset_end}")
        test_qnn(4, 2, 3, [total_shots], args.num_trials, False, dataset_start, dataset_end)
    elif args.task_name == "HWEA_RYRZ_CZ":
        print("Processing HWEA_RYRZ_CZ task")
        main(args.task_name, args.n_qubits, args.d, args.qubit_constrain, [args.total_shots], args.num_trials, args.with_classical_shadows)
    elif args.task_name == "3local":
        print("Processing 3local task")
        main(args.task_name, args.n_qubits, args.d, args.qubit_constrain, [args.total_shots], args.num_trials, args.with_classical_shadows)