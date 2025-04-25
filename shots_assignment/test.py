import pdb
import os
import copy

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePerth
from qiskit.quantum_info import DensityMatrix

# backend = AerSimulator.from_backend(FakePerth())
# backend_noisefree = AerSimulator(method='density_matrix', max_shot_size=None)

from qiskit.circuit.library import QFT
from qiskit import transpile
# from pgmpy.inference import VariableElimination
from qiskit.dagcircuit.dagnode import DAGOutNode, DAGInNode, DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
import struct
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography

from pgmQC.utils.state_transform import density_to_paulistring
from pgmQC.utils import plot_ps
from pgmQC.model import build_pgm, build_markov_networks, MarkovNetworkBuilder
from pgmQC.subcircuit_backend.density_matrix import DensityMatrixSimulator
from pgmQC.subcircuit_backend.noise_model import NoiseModelSimulator
from pgmQC.subcircuit_backend.circuit_fragment import CircuitFragment
from pgmQC.postprocessing_backend.cotengra_quimb import contract_tensors_and_plot
from collections import deque
import networkx as nx
import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp

import networkx as nx
import pgmQC.test_dataset.benchmarks as bm
import math

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
    elif name == 'qaoa':
        circuit = get_QAOA_circuit(n_qubits)
        circuit = transpile(circuit, basis_gates=['cx', 'u3', 'rzz'])
        return circuit
    elif name == 'vqe':
        circuit = get_VQE_circuit(n_qubits)
        circuit = transpile(circuit, basis_gates=['cx', 'u3'])
        return circuit
    elif name == 'qec':
        circuit = get_QEC_circuit(n_qubits)
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


def main(task_name, n_qubits, with_noise):
    circuit = generate_circuit_by_name(task_name, n_qubits)

    dag = circuit_to_dag(circuit)
    
    markov_net_builder = MarkovNetworkBuilder(dag, noise_inject=with_noise)
    
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
    
    fragment1 = CircuitFragment(dag, enable, markov_net_builder)
    fragment2 = CircuitFragment(dag, disable, markov_net_builder)
    cov1 = fragment1.analyze_variance(uncontracted_nodes1)
    cov2 = fragment2.analyze_variance(uncontracted_nodes2)
    n = circuit.num_qubits
    f3_variables = list((set(uncontracted_nodes1) | set(uncontracted_nodes2)) -(set(uncontracted_nodes1) & set(uncontracted_nodes2)))
    

    var_list = list(set(uncontracted_nodes1 + uncontracted_nodes2 + f3_variables))

    # contract_tensors_and_plot(path, [tensor1_noisefree, tensor2_noisefree], "arrays_noisefree.pkl")
    # contract_tensors_and_plot(path, [tensor1, tensor2], "arrays.pkl")


if __name__ == '__main__':
    main('qft', 2, False)
    # for n_qubits in [2, 3, 4, 5, 6]:
    #     main('qaoa', n_qubits, False)
    # for n_qubits in [10, 12, 14, 16]:
    #     main('vqe', n_qubits, False)
    # main('vqe', 4, False)
    # main('ghz', 103, False)
    # main('sycamore', 16, False)
    # main('supremacy', 169, False)
    # main('erdos', 57, False)
    # main('wstate', 60, False)