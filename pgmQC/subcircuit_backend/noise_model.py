import numpy as np
import pdb
import os

from qiskit.circuit.library import IQP
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp, random_hermitian
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakePerth, FakeManilaV2

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix

from pgmQC.model.markov_network_builder import MarkovNetworkBuilder
from qiskit.quantum_info import Pauli, SparsePauliOp, DensityMatrix, random_clifford, Operator
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from pgmQC.utils import build_circuit_from_subdag, plot_ps

from pgmQC.utils.state_transform import density_to_paulistring

class NoiseModelSimulator:
    def __init__(self, dag: DAGCircuit, enable, markov_network_builder : MarkovNetworkBuilder):
        self.dag = dag
        self.enable = enable
        self.markov_network_builder = markov_network_builder
        self.input_edges, self.output_edges = self.get_input_and_output_subdag_edges(dag, enable)
        self.circuit = build_circuit_from_subdag(dag, enable)
        self.tensor = None
        self.backend_name = 'fakebackend'
        # self.backend_name = 'realbackend'
        
        self.n_qubits = max([self.circuit.find_bit(qubit).index for qubit in self.dag.qubits]) + 1
        self.backend = FakeManilaV2()

    # get the input and output edges of a subdag
    def get_input_and_output_subdag_edges(self, dag, enable):
        input_edges = []
        output_edges = []
        for edge in dag.edges():
            if not enable[edge[0]] and enable[edge[1]]:
                input_edges.append(edge)
            if enable[edge[0]] and not enable[edge[1]]:
                output_edges.append(edge)
        return input_edges, output_edges
    
    # run the simulation
    def run(self, uncontracted_nodes):
        # prepare input state
        n_qubits = self.n_qubits
        
        input_qargs = []
        input_qargs_index = []
        for edge in self.input_edges:
            input_qargs.append(edge[-1])
            input_qargs_index.append(self.circuit.find_bit(edge[-1]).index)
            
        output_qargs = []
        output_qargs_index = []
        for edge in self.output_edges:
            output_qargs.append(edge[-1])
            output_qargs_index.append(self.circuit.find_bit(edge[-1]).index)
        output_qargs_index = list(reversed(output_qargs_index))
        
        input_states = ['']
        for edge in self.input_edges:
            qarg = edge[-1]
            new_input_states = []
            for input_state in input_states:
                for symbol in ['0', '1', '+', 'i']:
                    new_input_states.append(input_state + symbol)
            input_states = new_input_states
        
        
        output_states = ['']
        for edge in self.output_edges:
            qarg = edge[-1]
            new_output_states = []
            for output_state in output_states:
                for symbol in ['I', 'X', 'Y', 'Z']:
                    new_output_states.append(output_state + symbol)
            output_states = new_output_states
        
        circuit_list = []
        for input_state in input_states:
            state_initialization = QuantumCircuit(self.dag.qubits)
            for symbol, input_qarg in zip(input_state, input_qargs):
                if symbol == '0':
                    pass
                elif symbol == '1':
                    state_initialization.x(input_qarg)
                elif symbol == '+':
                    state_initialization.h(input_qarg)
                elif symbol == 'i':
                    state_initialization.h(input_qarg)
                    state_initialization.s(input_qarg)
            for ins in self.circuit:
                state_initialization.append(ins)
            circuit_list.append(state_initialization)
        
        pm = generate_preset_pass_manager(optimization_level=1, backend=self.backend)

        # for each input state, use qiskit backend to calculate the output density matrix
        pubs = []
        output_factors = []
        print("preparing jobs...")
        for circuit in circuit_list:
            for observable_string in output_states:
                observable = SparsePauliOp(observable_string[::-1]) # reverse the dimension, neccessary!
                isa_circuit = pm.run(circuit)
                isa_obs = observable.apply_layout(isa_circuit.layout)
                pubs.append((isa_circuit, isa_obs))
        
        print("End preparing jobs.")


        
        estimator = Estimator(self.backend)
        job = estimator.run(pubs)
        job_result = job.result()
        for idx in range(len(pubs)):
            pub_result = job_result[idx]
            print(f">>> Expectation values for PUB {idx}: {pub_result.data.evs}")
            print(f">>> Standard errors for PUB {idx}: {pub_result.data.stds}")
            output_factors.append(np.array(pub_result.data.evs))
        
        original_factor = np.array(output_factors)
        assert original_factor.shape == (4 ** (len(input_qargs) + len(output_qargs)),)
        
        # transform the input states from ['0', '1', '+', 'i'] to ['I', 'X', 'Y', 'Z']
        original_factor.reshape([4 for _ in range(len(input_qargs) + len(output_qargs))])
        cnt = 0
        total = len(input_qargs) + len(output_qargs)

        for i in range(len(input_qargs)):
            original_factor = original_factor.reshape(4 ** cnt, 4, 4 ** (total - cnt - 1))
            # I <= |0><0|+|1><1|
            # X <= 2|+><+|-I
            # Y <= 2|i><i|-I
            # Z <= |0><0|-|1><1|
            original_factor[:, 0, :], original_factor[:, 1, :], original_factor[:, 2, :], original_factor[:, 3, :] = \
                original_factor[:, 0, :] + original_factor[:, 1, :],\
                original_factor[:, 2, :] * 2 - original_factor[:, 0, :] - original_factor[:, 1, :],\
                original_factor[:, 3, :] * 2 - original_factor[:, 0, :] - original_factor[:, 1, :],\
                original_factor[:, 0, :] - original_factor[:, 1, :]
            cnt = cnt + 1
        
        permutation = [-1 for _ in range(len(input_qargs) + len(output_qargs))]
        for i, input_edge in enumerate(self.input_edges):
            var = self.markov_network_builder.get_var_name(input_edge)
            index = uncontracted_nodes.index(var)
            permutation[index] = i
        for i, output_edge in enumerate(self.output_edges):
            var = self.markov_network_builder.get_var_name(output_edge)
            index = uncontracted_nodes.index(var)
            permutation[index] = i + len(self.input_edges)
        
        assert all([p != -1 for p in permutation])
        
        factor = original_factor.reshape([4 for _ in range(len(input_qargs) + len(output_qargs))])
        factor = factor.transpose(permutation)
        self.tensor = factor.reshape(-1)
        return self.tensor
    
    def draw_circuit(self, filename):
        self.circuit.draw(output='mpl', filename=filename)
    
    def plot_tensor(self, filename):
        plot_ps(self.tensor, filename)
    
    def plot_all(self, directory):
        self.draw_circuit(os.path.join(directory, 'circuit.png'))
        self.plot_tensor(os.path.join(directory, f'tensor_{self.backend_name}.png'))