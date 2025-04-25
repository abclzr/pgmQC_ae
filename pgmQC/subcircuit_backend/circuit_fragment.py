import numpy as np
import pdb
import os

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix

from pgmQC.model.markov_network_builder import MarkovNetworkBuilder
from qiskit.quantum_info import Pauli, SparsePauliOp, DensityMatrix, Statevector, random_clifford, Operator
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from pgmQC.utils import build_circuit_from_subdag, plot_ps

from pgmQC.utils.state_transform import density_to_paulistring
from pgmQC.subcircuit_backend.independent_experiment import IndependentExperiment
from pgmQC.subcircuit_backend.corelated_random_variables import CorelatedRandomVariables, RandomVariable

class CircuitFragment:
    def __init__(self, dag: DAGCircuit, enable, markov_network_builder : MarkovNetworkBuilder):
        self.dag = dag
        self.enable = enable
        self.markov_network_builder = markov_network_builder
        self.input_edges, self.output_edges = self.get_input_and_output_subdag_edges(dag, enable)
        self.circuit = build_circuit_from_subdag(dag, enable)
        self.tensor = None
        self.backend_name = 'sampler'
        self.experiment_list = []
        self.experiment_dict = {}
    
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
    
    # generate_state(['+', 'i'], 2) returns ['++', '+i', 'i+', 'ii']
    # use this function to generate all possible input states and output states
    def generate_state(self, symbol_list, num_edges):
        states = ['']
        for edge in num_edges:
            qarg = edge[-1]
            new_states = []
            for input_state in states:
                for symbol in symbol_list:
                    new_states.append(input_state + symbol)
            states = new_states
        return states
    
    """ For a given output measurement basis, create the eigenvalues for each eigenbasis.
    
        Example:
            if output measurement basis is 'XZI',
            the eigenvalues are [1, 1, -1, -1, 1, 1, -1, -1]
            for bitstrings representing the bases are [000, 001, 010, 011, 100, 101, 110, 111]
    """
    def create_eigenvalues(self, output_state):
        ret = 1
        for symbol in output_state:
            if symbol == 'X' or symbol == 'Y' or symbol == 'Z':
                ret = np.kron(ret, [1, -1])
            elif symbol == 'I':
                ret = np.kron(ret, [1, 1])
            else:
                raise ValueError(f'Invalid symbol {symbol}')
        return ret
    
    # analyze the variance of within the circuit fragment.
    def analyze_variance(self, uncontracted_nodes):
        # prepare input state
        n_qubits = max([self.circuit.find_bit(qubit).index for qubit in self.dag.qubits]) + 1
        input_qargs = []
        input_qargs_index = []
        for edge in self.input_edges:
            # egde is a 3-tuple: (input_node, output_node, qubit), so edge[-1] is the qubit
            input_qargs.append(edge[-1])
            input_qargs_index.append(self.circuit.find_bit(edge[-1]).index)
            
        output_qargs = []
        output_qargs_index = []
        for edge in self.output_edges:
            output_qargs.append(edge[-1])
            output_qargs_index.append(self.circuit.find_bit(edge[-1]).index)
        output_qargs_index = list(reversed(output_qargs_index))
        
        input_states = self.generate_state(['0', '1', '+', 'i'], self.input_edges)
        output_states = self.generate_state(['X', 'Y', 'Z'], self.output_edges)
        # for each input state, use qiskit backend to calculate the output state vector and its probability distribution on the bitstrings
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
            
            statevec = Statevector.from_instruction(state_initialization).evolve(self.circuit)
            output_factor = []
            for output_state in output_states:
                measurement_transformation = QuantumCircuit(self.dag.qubits)
                for symbol, output_qargs in zip(output_state, output_qargs):
                    if symbol == 'X':
                        measurement_transformation.h(output_qargs)
                    elif symbol == 'Y':
                        measurement_transformation.sdg(output_qargs)
                        measurement_transformation.h(output_qargs)
                    elif symbol == 'Z':
                        pass
                final_statevec = statevec.evolve(measurement_transformation)
                pdb.set_trace()
                
                """ reverse the output_qargs_index to match the order of the bitstrings. Here's why:
                    in qiskit, the qubit index is from the least significant bit to the most significant bit
                    here, output_qargs_index is from the most significant bit to the least significant bit
                    so we need to reverse the order of output_qargs_index
                """
                self.experiment_dict[input_state + output_state] = \
                    IndependentExperiment(input_state, output_state, final_statevec.probabilities(output_qargs_index[::-1]))
        
        # adding symbol 'I' to the output states ['X', 'Y', 'Z']
        new_output_states = self.generate_state(['I', 'X', 'Y', 'Z'], self.output_edges)
        for input_state in input_states:
            for output_state in new_output_states:
                output_state_to_lookup = output_state.replace('I', 'Z')
                exp = self.experiment_dict[input_state + output_state_to_lookup]
                assert type(exp) == IndependentExperiment
                # calculate the eigenvalues, mean and variance of the eigenvalues.
                eigenvalues = self.create_eigenvalues(output_state)
                mean = np.sum(eigenvalues * exp.prob)
                variance = 1 - mean ** 2
                rv = RandomVariable(mean, variance, input_state + output_state)
                rv.assign_eigenvalues(eigenvalues)
                exp.add_random_variable(rv)
        
        covariance_table = {}
        covariance_matrix = np.zeros(4 ** (len(input_qargs) + len(output_qargs)), 4 ** (len(input_qargs) + len(output_qargs)))
        # calculate the covariance between the random variables from the same experiment
        for key, value in self.experiment_dict.items():
            value.deal_with_covariance(covariance_matrix)
            covariance_table.update(value.covariance)
        
        """ transform the input states from ['0', '1', '+', 'i'] to ['I', 'X', 'Y', 'Z']
            I = |0><0|+|1><1|
            X = 2|+><+|-I
            Y = 2|i><i|-I
            Z = |0><0|-|1><1|
            
            |I|   | 1  1  0  0| |0|
            |X| = |-1 -1  2  0| |1|
            |Y|   |-1 -1  0  2| |+|
            |Z|   | 1 -1  0  0| |i|
        """
        cnt = 0
        total = len(input_qargs) + len(output_qargs)
        new_input_states = input_states
        H = 1
        for i in range(len(input_qargs)):
            H = np.kron(H, np.array([[1, 1, 0, 0], [-1, -1, 2, 0], [-1, -1, 0, 2], [1, -1, 0, 0]]))
        for i in range(len(output_qargs)):
            H = np.kron(H, np.eye(4))
        covariance_matrix = H @ covariance_matrix @ H.T
        
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
        return covariance_matrix
    
    def draw_circuit(self, filename):
        self.circuit.draw(output='mpl', filename=filename)
    
    def plot_tensor(self, filename):
        plot_ps(self.tensor, filename)
    
    def plot_all(self, directory):
        self.draw_circuit(os.path.join(directory, 'circuit.png'))
        self.plot_tensor(os.path.join(directory, f'tensor_{self.backend_name}.png'))