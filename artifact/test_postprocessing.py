import pdb
import os
import pickle
from tqdm import tqdm

from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit.circuit.library import QFT
from qiskit import transpile
# from pgmpy.inference import VariableElimination
from qiskit.dagcircuit.dagnode import DAGOutNode, DAGInNode, DAGOpNode
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagcircuit import DAGCircuit
import struct

from pgmQC.utils import qiskit_wire_cut
from pgmQC.model import MarkovNetworkBuilder, TensorNetworkBuilder
from pgmQC.postprocessing_backend.cotengra_quimb import contract_tensor_from_pgmpy, quimb_contraction
from collections import deque
import networkx as nx
import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_addon_cutting.instructions import Move

import networkx as nx
import pgmQC.test_dataset.benchmarks as bm
import math
np.random.seed(42)

DEBUG=False

def get_QAOA_circuit(num_nodes):
    w = np.full((num_nodes, num_nodes), 1)    
    num_qubits = num_nodes
    def get_operator(weight_matrix: np.ndarray, n_qubits: int) -> tuple[QuantumCircuit, SparsePauliOp, float]:
        r"""Generate Hamiltonian for the graph partitioning
        Notes:
            Goals:
                1 Separate the vertices into two set of the same size.
                2 Make sure the number of edges between the two set is minimized.
            Hamiltonian:
                H = H_A + H_B
                H_A = sum\_{(i,j)\in E}{(1-ZiZj)/2}
                H_B = (sum_{i}{Zi})^2 = sum_{i}{Zi^2}+sum_{i!=j}{ZiZj}
                H_A is for achieving goal 2 and H_B is for achieving goal 1.
        Args:
            weight_matrix: Adjacency matrix.
        Returns:
            Operator for the Hamiltonian
            A constant shift for the obj function.
        """
        num_nodes = len(weight_matrix)
        pauli_list = []
        coeffs = []
        shift = 0
        circuit = QuantumCircuit(n_qubits)

        for i in range(num_nodes):
            for j in range(i):
                if weight_matrix[i, j] != 0:
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append(Pauli((z_p, x_p)))
                    coeffs.append(-0.5)
                    shift += 0.5
                    circuit.rzz(-0.5, i, j)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append(Pauli((z_p, x_p)))
                    coeffs.append(1.0)
                    circuit.rzz(1, i, j)
                else:
                    shift += 1

        return circuit, SparsePauliOp(pauli_list, coeffs=coeffs), shift

    circuit, qubit_op, offset = get_operator(w, num_qubits)
    
    return circuit

def get_VQE_circuit(n_qubits):
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
    mapper = JordanWignerMapper()

    ansatz = UCCSD(
        n_qubits // 2,
        (1, 1),
        mapper,
        initial_state=HartreeFock(
            n_qubits // 2,
            (1, 1),
            mapper,
        ),
    )

    params_list = np.random.rand(len(ansatz.parameters)) * np.pi * 2 - np.pi
    print(params_list)
    params_bind = {x:y for x, y in zip(ansatz.parameters, params_list)}
    circuit = ansatz.assign_parameters(params_bind)
    return circuit

def get_QEC_circuit(n_qubits):
    circuit = QuantumCircuit(5)
    circuit.x(0)
    # encode
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    # decode
    circuit.cx(0, 3)
    circuit.cx(1, 3)
    circuit.cx(1, 4)
    circuit.cx(2, 4)
    return circuit

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

def get_nlocal(n_qubits, d=1, qubit_constrain=4, pruning_ratio=0., n_local=3):
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
    
    def do_nlocal_entanglement(circ, start, n_local):
        for j in range(1, n_local):
            circ.cz(start, start + j)
    
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
        for j in range(n_qubits - (n_local - 1)):
            do_nlocal_entanglement(circuit, j, n_local)
        j = 0
        while j < n_qubits - (n_local - 1):
            if j in cut_start_position_logical:
                for k in range(n_local - 1):
                    if j + k + n_local - 1 < n_qubits:
                        circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j + k], map_to_physical_qubits[j + k]+n_local-1])
                for k in range(n_local - 1):
                    if j + k + n_local - 1 < n_qubits:
                        do_nlocal_entanglement(circuit_for_qiskit_cut, map_to_physical_qubits[j + k]+n_local-1, n_local)
                for k in range(n_local - 1):
                    if j + k + n_local - 1 < n_qubits:
                        circuit_for_qiskit_cut.append(Move(), [map_to_physical_qubits[j + k]+n_local-1, map_to_physical_qubits[j + k]])
                j = j + n_local-1
            else:
                do_nlocal_entanglement(circuit_for_qiskit_cut, map_to_physical_qubits[j], n_local)
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
            if np.random.rand() > pruning_ratio:
                circuit.rz(b, j)
                circuit_for_qiskit_cut.rz(b, map_to_physical_qubits[j])
        ob_str[map_to_physical_qubits[j]] = "Z"
    observable = SparsePauliOp(["".join(['Z'] * n_qubits)])
    observable_expanded = SparsePauliOp(["".join(reversed(ob_str))])
    
    return circuit, circuit_for_qiskit_cut, observable, observable_expanded, partition_label

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
        circuit = transpile(circuit, basis_gates=['cx', 'h', 's', 'sdg', 'rz'])
        return circuit
    elif name == 'qaoa':
        circuit = get_QAOA_circuit(n_qubits)
        circuit = transpile(circuit, basis_gates=['cx', 'h', 's', 'sdg', 'rz'])
        return circuit
    elif name == 'vqe':
        circuit = get_VQE_circuit(n_qubits)
        circuit = transpile(circuit, basis_gates=['cx', 'h', 's', 'sdg', 'rz'])
        return circuit
    elif name == 'qec':
        circuit = get_QEC_circuit(n_qubits)
        return circuit
    elif name == 'ghz':
        circuit = bm.ghz(num_qubits=n_qubits)
        circuit = transpile(circuit, basis_gates=['cx', 'h', 's', 'sdg', 'rz'])
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
        
        # circuit = transpile(circuit, basis_gates=['cx', 'cz', 'u3', 't', 'x', 'y'])
        circuit = transpile(circuit, basis_gates=['cx', 'h', 's', 'sdg', 'rz'])
        return circuit
    elif name == 'sycamore':
        grid_length = int(math.sqrt(n_qubits))
        assert (
            grid_length**2 == n_qubits
        ), "Sycamore is defined for n*n square circuits"
        circuit = bm.gen_sycamore(grid_length, grid_length, depth * 8, regname="q")
        # circuit = transpile(circuit, basis_gates=['cx', 'cz', 'u3', 't', 'x', 'y'])
        circuit = transpile(circuit, basis_gates=['cx', 'h', 's', 'sdg', 'rz'])
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

        # circuit = transpile(circuit, basis_gates=['cx', 'u3', 'rx', 'rz'])
        circuit = transpile(circuit, basis_gates=['cx', 'h', 's', 'sdg', 'rz'])
        return circuit
    elif name == 'HWEA':
        return get_HWEA(n_qubits)
    elif name == 'nlocal':
        return get_nlocal(n_qubits)

def profile_circuit(circ):
    total = 0
    for gate, number in circ.count_ops().items():
        print(f'{gate} : {number}')
        total += number
    print(f'total: {total}')
    Tgates = 0
    for ins in circ:
        if ins.operation.name == 'rz':
            if ins.operation.params[0] % (np.pi/2) != 0:
                Tgates += 1
    print(f'T gates: {Tgates}')

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


def generate_final_tensor_description(path, uncontracted_nodes_list):
    filename = path + 'description.txt'
    # remove duplicates in uncontracted_nodes_list
    var_list = list(set([item for sublist in uncontracted_nodes_list for item in sublist]))
    num_variables = len(var_list)
    
    with open(filename, "w") as file:
        file.write(str(num_variables) + '\n')
        for var in var_list:
            file.write(var + " ")
        file.write('\n')
        for var in var_list:
            file.write('4' + " ")
        file.write('\n')
        
        file.write(str(len(uncontracted_nodes_list) + 1) + '\n')
        for f_variables in uncontracted_nodes_list + [[]]:
            for var in f_variables:
                file.write(var + " ")
            file.write('\n')
            
def pickle_store(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def get_backend():
    return AerSimulator(method='statevector')
    # return FakeManilaV2()

def main(task_name, n_qubits, d, qubit_constrain):
    # circuit = generate_circuit_by_name(task_name, n_qubits)
    if True:
        if task_name == 'HWEA_RYRZ_CZ':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_HWEA(n_qubits, d, qubit_constrain)
        elif task_name == 'pruned_0.5_HWEA3_RYRZ_CZ' or task_name == 'pruned_0.5_HWEA6_RYRZ_CZ':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_HWEA(n_qubits, d, qubit_constrain, 0.5)
        elif task_name == 'pruned_0.9_HWEA3_RYRZ_CZ' or task_name == 'pruned_0.9_HWEA6_RYRZ_CZ':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_HWEA(n_qubits, d, qubit_constrain, 0.9)
        elif task_name == 'HWEA_RY_CX':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = bm.get_HWEA_Michele_Grossi(n_qubits, d, qubit_constrain)
        elif task_name == 'pruned_0.5_HWEA_RY_CX':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = bm.get_HWEA_Michele_Grossi(n_qubits, d, qubit_constrain, 0.5)
        elif task_name == 'pruned_0.9_HWEA_RY_CX':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = bm.get_HWEA_Michele_Grossi(n_qubits, d, qubit_constrain, 0.9)
        elif task_name == 'aqft_3':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_aqft(n_qubits, 3, qubit_constrain)
        elif task_name == 'aqft_6':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_aqft(n_qubits, 6, qubit_constrain)
        elif task_name == 'aqft_7':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_aqft(n_qubits, 7, qubit_constrain)
        elif task_name == '3local':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 3)
        elif task_name == '4local':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0., 4)
        elif task_name == 'pruned_0.5_4local':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0.5, 4)
        elif task_name == 'pruned_0.9_4local':
            circuit_for_pgmQC, circuit, observable, observable_expanded, partition_label = get_nlocal(n_qubits, d, qubit_constrain, 0.9, 4)
        else:
            raise ValueError(f'Unknown task name: {task_name}')
            
    # circuit_for_pgmQC.draw("mpl", scale=0.8, filename='original_circuit.png')
    # circuit.draw("mpl", scale=0.8, filename='circuit.png')
    
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
    def sparsity(arr, tol=1e-10):
        close_to_zero = np.abs(arr) < tol
        # Calculate the percentage
        sp = 1 - np.sum(close_to_zero) / arr.size
        return sp
    pgmQC_total_experiments = 0
    remained_experiments_ratio = []
    
    uncontracted_nodes_list = []
    for label, tensor_network_builder in tensor_network_builders.items():
        tensor_net, uncontracted_nodes = tensor_network_builder.build(probability_all_zero_state=(task_name.startswith('aqft')))
        contracted_tensor = contract_tensor_from_pgmpy(tensor_net, uncontracted_nodes)
        arrays.append(contracted_tensor.data)
        inds.append(contracted_tensor.inds)
        true_false_tensor = contract_tensor_from_pgmpy(tensor_network_builder.true_false_network, uncontracted_nodes)
        print(sparsity(true_false_tensor.data))
        num_subcircuit_experiments = tensor_network_builder.num_experiments(true_false_tensor.data, true_false_tensor.inds)
        pgmQC_total_experiments += num_subcircuit_experiments
        remained_experiments_ratio.append(num_subcircuit_experiments\
            / (4 ** len(tensor_network_builder.input_dag_nodes)) * (3 ** len(tensor_network_builder.output_dag_nodes)))
        
        generate_uncontracted_tensors(tensor_net, uncontracted_nodes, f'../dataset/{task_name}_{n_qubits}/{label}/')
        uncontracted_nodes_list.append(uncontracted_nodes)
    generate_final_tensor_description(f'../dataset/{task_name}_{n_qubits}/', uncontracted_nodes_list)
    print(f'exact expval: {quimb_contraction(arrays, inds)}')


def cut_in_middle(task_name, n_qubits, with_noise=False):
    circuit = generate_circuit_by_name(task_name, n_qubits)

    dag = circuit_to_dag(circuit)
    
    markov_net_builder = MarkovNetworkBuilder(dag, noise_inject=with_noise)
    markov_net, uncontracted_nodes = markov_net_builder.build()
    
    nodes = list(dag.nodes())
    number_nodes = len(nodes)
    
    starts = []
    # perform a cut
    for node in nodes:
        if isinstance(node, DAGInNode):
            starts.append(node)

    enable = bfs(starts, number_nodes // 2, dag)
    # disable = {key: not value for key, value in enable.items()}
    disable = {key: not isinstance(key, DAGOutNode) and not value for key, value in enable.items()}
    
    markov_net1, uncontracted_nodes1 = markov_net_builder.build(enable)
    markov_net2, uncontracted_nodes2 = markov_net_builder.build(disable)
    
    # backend1_noisefree = DensityMatrixSimulator(dag, enable, markov_net_builder)
    # backend2_noisefree = DensityMatrixSimulator(dag, disable, markov_net_builder)

    
    # tensor1_noisefree = backend1_noisefree.run(uncontracted_nodes1)
    # tensor2_noisefree = backend2_noisefree.run(uncontracted_nodes2)
    
    # backend1 = NoiseModelSimulator(dag, enable, markov_net_builder)
    # backend2 = NoiseModelSimulator(dag, disable, markov_net_builder)
    
    # tensor1 = backend1.run(uncontracted_nodes1)
    # tensor2 = backend2.run(uncontracted_nodes2)
    
    n = circuit.num_qubits
    generate_uncontracted_tensors(markov_net1, uncontracted_nodes1, f'../dataset/{task_name}_{n_qubits}/subcircuit_0/')
    # backend1.plot_all(directory=f'../dataset/qft_{n}_subgraph1/')
    generate_uncontracted_tensors(markov_net2, uncontracted_nodes2, f'../dataset/{task_name}_{n_qubits}/subcircuit_1/')
    # backend2.plot_all(directory=f'../dataset/qft_{n}_subgraph2/')
    
    f3_variables = list((set(uncontracted_nodes1) | set(uncontracted_nodes2)) -(set(uncontracted_nodes1) & set(uncontracted_nodes2)))
    

    var_list = list(set(uncontracted_nodes1 + uncontracted_nodes2 + f3_variables))

    num_variables = len(var_list)
    path = f'../dataset/{task_name}_{n_qubits}/'
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    filename = path + 'description.txt'
    if n < 8:
        circuit.draw(output='mpl', filename=path+'circuit.png')
        dag.draw(filename=path+'dag.png')

    with open(filename, "w") as file:
        file.write(str(num_variables) + '\n')
        for var in var_list:
            file.write(var + " ")
        file.write('\n')
        for var in var_list:
            file.write('4' + " ")
        file.write('\n')
        
        file.write(str(3) + '\n')
        for f_variables in [uncontracted_nodes1, uncontracted_nodes2, f3_variables]:
            for var in f_variables:
                file.write(var + " ")
            file.write('\n')
    
    # contract_tensors_and_plot(path, [tensor1_noisefree, tensor2_noisefree], "arrays_noisefree.pkl")
    # contract_tensors_and_plot(path, [tensor1, tensor2], "arrays.pkl")



if __name__ == '__main__':
    for n_qubits in range(2, 8):
        cut_in_middle('qft', n_qubits, with_noise=False)
    for n_qubits in [50, 100, 150, 200]:
        # main('aqft_3', n_qubits, 1, 20)
        # main('aqft_6', n_qubits, 1, 20)
        main('aqft_7', n_qubits, 1, 20)
        main('3local', n_qubits, 1, 20)