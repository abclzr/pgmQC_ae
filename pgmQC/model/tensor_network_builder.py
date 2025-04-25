import numpy as np
import pdb
import copy

from qiskit import dagcircuit, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import CircuitInstruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from qiskit.quantum_info import Pauli, SparsePauliOp, DensityMatrix, random_clifford
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit_ibm_runtime import SamplerV2, Batch
from pgmQC.utils.setting import I, X, Y, Z

class TensorNetworkBuilder:
    def __init__(self, dag: DAGCircuit, prefix: str):
        self.dag = dag
        self.dag.draw(filename="dag.png")
        self.pauli_index_lookup = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        self.nodes = list(self.dag.nodes())
        self.vars = list(self.dag.edges())
        self.var_name_lookup = {}
        self.is_this_qpd_1q_an_input_edge = {}
        self.input_dag_nodes = []
        self.input_dag_nodes_index = {}
        self.input_edge_names = []
        self.output_dag_nodes = []
        self.output_dag_nodes_index = {}
        self.output_edge_names = []
        self.closed_dag_out_nodes = []
        self.closed_dag_out_nodes_Pauli = {}
        self.nodes_already_removed = []
        # for i, var in enumerate(self.vars):
        #     if isinstance(var[0], dagcircuit.DAGInNode) and isinstance(var[1], dagcircuit.DAGOutNode):
        #         self.var_name_lookup[var] = 'this_edge_should_not_exists'
        #     elif isinstance(var[0], dagcircuit.DAGInNode) and isinstance(var[1], dagcircuit.DAGOpNode) and var[1].op.name == 'qpd_1q':
        #         self.var_name_lookup[var] = 'this_edge_should_not_exists'
        #     elif isinstance(var[1], dagcircuit.DAGOutNode) and isinstance(var[0], dagcircuit.DAGOpNode):
        #         if var[0].op.name == 'qpd_1q':
        #             self.var_name_lookup[var] = 'this_edge_should_not_exists'
        #         else:
        #             self.var_name_lookup[var] = f'{prefix}_{i}'
        #             self.closed_dag_out_nodes.append(var[1])
        #     elif isinstance(var[0], dagcircuit.DAGOpNode) and isinstance(var[1], dagcircuit.DAGOpNode):
        #         if var[0].op.name == 'qpd_1q' and var[1].op.name == 'qpd_1q':
        #             self.var_name_lookup[var] = 'this_edge_should_not_exists'
        #         elif var[0].op.name == 'qpd_1q' and isinstance(var[1], dagcircuit.DAGOpNode):
        #             self.var_name_lookup[var] = var[0].op.label
        #             self.input_edge_names.append(var[0].op.label)
        #             self.input_dag_nodes_index[var[0].op.label] = len(self.input_dag_nodes)
        #             self.input_dag_nodes.append(var[0])
        #             self.is_this_qpd_1q_an_input_edge[var[0]] = True
        #         elif var[1].op.name == 'qpd_1q' and isinstance(var[0], dagcircuit.DAGOpNode):
        #             self.var_name_lookup[var] = var[1].op.label
        #             self.output_edge_names.append(var[1].op.label)
        #             self.output_dag_nodes_index[var[1].op.label] = len(self.output_dag_nodes)
        #             self.output_dag_nodes.append(var[1])
        #             self.is_this_qpd_1q_an_input_edge[var[1]] = False
        #         else:
        #             self.var_name_lookup[var] = f'{prefix}_{i}'

        #     else:
        #         self.var_name_lookup[var] = f'{prefix}_{i}'
        for i, var in enumerate(self.vars):
            if isinstance(var[0], dagcircuit.DAGOpNode) and isinstance(var[1], dagcircuit.DAGOpNode):
                if var[0].op.name == 'qpd_1q' and var[1].op.name == 'qpd_1q':
                    self.var_name_lookup[var] = 'this_edge_should_not_exists'
                    self.nodes_already_removed.append(var[0])
                    self.nodes_already_removed.append(var[1])
                    self.is_this_qpd_1q_an_input_edge[var[0]] = False
                    self.is_this_qpd_1q_an_input_edge[var[1]] = True
        for i, var in enumerate(self.vars):
            if isinstance(var[0], dagcircuit.DAGInNode) and isinstance(var[1], dagcircuit.DAGOutNode):
                self.var_name_lookup[var] = 'this_edge_should_not_exists'
            elif isinstance(var[0], dagcircuit.DAGInNode) and isinstance(var[1], dagcircuit.DAGOpNode) and var[1].op.name == 'qpd_1q':
                if var[1] not in self.nodes_already_removed:
                    self.var_name_lookup[var] = 'this_edge_should_not_exists'
                else:
                    self.var_name_lookup[var] = var[1].op.label
                    self.output_edge_names.append(var[1].op.label)
                    self.output_dag_nodes_index[var[1].op.label] = len(self.output_dag_nodes)
                    self.output_dag_nodes.append(var[1])
                    self.is_this_qpd_1q_an_input_edge[var[1]] = False
            elif isinstance(var[1], dagcircuit.DAGOutNode) and isinstance(var[0], dagcircuit.DAGOpNode):
                if var[0].op.name == 'qpd_1q':
                    if var[0] not in self.nodes_already_removed:
                        self.var_name_lookup[var] = 'this_edge_should_not_exists'
                    else:
                        self.var_name_lookup[var] = var[0].op.label
                        self.input_edge_names.append(var[0].op.label)
                        self.input_dag_nodes_index[var[0].op.label] = len(self.input_dag_nodes)
                        self.input_dag_nodes.append(var[0])
                        self.is_this_qpd_1q_an_input_edge[var[0]] = True
                        self.closed_dag_out_nodes.append(var[1])
                else:
                    self.var_name_lookup[var] = f'{prefix}_{i}'
                    self.closed_dag_out_nodes.append(var[1])
            elif isinstance(var[0], dagcircuit.DAGOpNode) and isinstance(var[1], dagcircuit.DAGOpNode):
                if var[0].op.name == 'qpd_1q' and var[1].op.name == 'qpd_1q':
                    self.var_name_lookup[var] = 'this_edge_should_not_exists'
                elif var[0].op.name == 'qpd_1q' and isinstance(var[1], dagcircuit.DAGOpNode):
                    self.var_name_lookup[var] = var[0].op.label
                    self.input_edge_names.append(var[0].op.label)
                    self.input_dag_nodes_index[var[0].op.label] = len(self.input_dag_nodes)
                    self.input_dag_nodes.append(var[0])
                    self.is_this_qpd_1q_an_input_edge[var[0]] = True
                elif var[1].op.name == 'qpd_1q' and isinstance(var[0], dagcircuit.DAGOpNode):
                    self.var_name_lookup[var] = var[1].op.label
                    self.output_edge_names.append(var[1].op.label)
                    self.output_dag_nodes_index[var[1].op.label] = len(self.output_dag_nodes)
                    self.output_dag_nodes.append(var[1])
                    self.is_this_qpd_1q_an_input_edge[var[1]] = False
                else:
                    self.var_name_lookup[var] = f'{prefix}_{i}'

            else:
                self.var_name_lookup[var] = f'{prefix}_{i}'
        self.non_zero_experiments = None

    # get the var name of an edge
    def get_var_name(self, edge):
        return self.var_name_lookup[edge]
    
    # build the markov network
    # enable: a dictionary with the key as the node and the value as a boolean
    #         if the value is True, the node shall be enabled.
    #         if the value is False, the node shall be disabled.
    #         if the value is None, the node shall be enabled.
    #         dagcircuit.DAGInNode shall always be enabled.var_name(edge)
    # dagcircuit.DAGOutNode shall never be enabled.
    def build(self, enable=None, observable=None, show_density_matrix=False, probability_all_zero_state=False):
        nodes = self.nodes
        vars = self.vars
        var_name_lookup = self.var_name_lookup
        pauli_index_lookup = self.pauli_index_lookup
        
        markov_net = MarkovNetwork()
        factors = set()
        self.true_false_network = MarkovNetwork()
        
        def connected(opnode):
            connected_vars = []
            for var in vars:
                if var[0] == opnode or var[1] == opnode:
                    connected_vars.append(var)
            return connected_vars
        
        for opnode in nodes:
            if enable != None and enable[opnode] == False:
                continue
            if isinstance(opnode, dagcircuit.DAGInNode):
                connected_vars = connected(opnode)
                if var_name_lookup[connected_vars[0]] == 'this_edge_should_not_exists':
                    continue
                factor = DiscreteFactor([var_name_lookup[connected_vars[0]]],
                        cardinality=[4],
                        values=[1, 0, 0, 1],
                        state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                )
                factors.add(factor)
            elif isinstance(opnode, dagcircuit.DAGOpNode):
                if opnode.op.name == 'qpd_1q':
                    continue
                # sort the connected variables based on the opnode.qargs
                connected_vars = connected(opnode)
                connected_vars = [(var, var[1] == opnode, opnode.qargs.index(var[2])) for var in connected_vars]
                connected_vars = sorted(connected_vars, key = lambda x: (not x[1], x[2]))
                connected_vars = [var[0] for var in connected_vars]
                
                # calculate the values in the factors
                
                # initialize the input variables state
                paulis = [ Pauli('I'), Pauli('X'), Pauli('Y'), Pauli('Z') ]
                strings = paulis

                n = opnode.op.num_qubits
                while n > 1:
                    new_strings = []
                    for a in strings:
                        for b in paulis:
                            new_strings.append(a.expand(b))
                    strings = new_strings
                    n = n-1

                # for every input variables state, get the output variables state
                values = []
                for string in strings:
                    sparse = SparsePauliOp.from_operator(DensityMatrix(string).evolve(opnode.op))
                    out_state_values = np.zeros(4 ** opnode.op.num_qubits)
                    for label, coeff in sparse.label_iter(): # type: ignore
                        index = 0
                        for pauli in reversed(label):
                            index = index * 4 + pauli_index_lookup[pauli]
                        out_state_values[index] = coeff
                    
                    values.append(out_state_values)

                # create the factor
                values = np.concatenate(values, axis=0).real
                factor = DiscreteFactor([var_name_lookup[var] for var in connected_vars],
                        cardinality=[4 for var in connected_vars],
                        values=values,
                        state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                )
                factors.add(factor)

            elif isinstance(opnode, dagcircuit.DAGOutNode):
                connected_vars = connected(opnode)
                if var_name_lookup[connected_vars[0]] == 'this_edge_should_not_exists':
                    continue
                if show_density_matrix:
                    continue
                if probability_all_zero_state:
                    value = [.5, 0, 0, .5]
                    self.closed_dag_out_nodes_Pauli[opnode] = 'Z'
                else:
                    if observable == None or observable[opnode.wire._index] == Pauli('Z'):
                        value = [0, 0, 0, 1.]
                        self.closed_dag_out_nodes_Pauli[opnode] = 'Z'
                    elif observable[opnode.wire._index] == Pauli('I'):
                        value = [1., 0, 0, 0]
                        self.closed_dag_out_nodes_Pauli[opnode] = 'I'
                    elif observable[opnode.wire._index] == Pauli('X'):
                        value = [0, 1., 0, 0]
                        self.closed_dag_out_nodes_Pauli[opnode] = 'X'
                    elif observable[opnode.wire._index] == Pauli('Y'):
                        value = [0, 0, 1., 0]
                        self.closed_dag_out_nodes_Pauli[opnode] = 'Y'
                factor = DiscreteFactor([var_name_lookup[connected_vars[0]]],
                    cardinality=[4],
                    values=value,
                    state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                )
                factors.add(factor)
            else:
                raise TypeError(opnode)

        for var in vars:
            if var_name_lookup[var] == 'this_edge_should_not_exists':
                continue
            if enable == None or enable[var[0]]:
                markov_net.add_node(var_name_lookup[var])
                self.true_false_network.add_node(var_name_lookup[var])
            elif enable[var[1]]:
                markov_net.add_node(var_name_lookup[var])
                self.true_false_network.add_node(var_name_lookup[var])


        for factor in factors:
            markov_net.add_factors(factor)
            self.true_false_network.add_factors(
                DiscreteFactor(factor.variables,
                    factor.cardinality,
                    np.where(factor.values != 0, 1, 0),
                    factor.state_names
                )
            )
            edges = [(factor.variables[i], factor.variables[j])
                    for i in range(len(factor.variables))
                    for j in range(i + 1, len(factor.variables))]
            markov_net.add_edges_from(ebunch=edges)
            self.true_false_network.add_edges_from(ebunch=edges)
        
        nodes_count = {node: 2 for node in markov_net.nodes}

        for factor in markov_net.factors:
            for node in factor.variables:
                nodes_count[node] -= 1
        
        uncontracted_nodes = []
        for node, count in nodes_count.items():
            if count == 1:
                uncontracted_nodes.append(node)
            else:
                assert count == 0

        for node in self.closed_dag_out_nodes:
            print(f'Measurement basis on qubit{node.wire._index}: {self.closed_dag_out_nodes_Pauli[node]}')
        
        return markov_net, uncontracted_nodes

    # generate_states(['+', 'i'], 2) returns ['++', '+i', 'i+', 'ii']
    # use this function to generate all possible input states and output states
    def generate_states(self, symbol_list, num_edges) -> list[str]:
        states = ['']
        for _ in range(num_edges):
            new_states = []
            for input_state in states:
                for symbol in symbol_list:
                    new_states.append(input_state + symbol)
            states = new_states
        return states
    
    def generate_all_experiments_naive(self) -> dict[str, QuantumCircuit]:
        input_states = self.generate_states(['0', '1', '+', 'i'], len(self.input_dag_nodes))
        output_bases = self.generate_states(['X', 'Y', 'Z'], len(self.output_dag_nodes))
        ret = {}
        for input_state in input_states:
            for output_basis in output_bases:
                circ = self.generate_subexperiment(input_state, output_basis)
                ret[input_state + output_basis] = circ
        return ret
    
    def generate_all_experiments(self) -> dict[str, QuantumCircuit]:
        if self.non_zero_experiments is not None:
            ret = {}
            for non_zero_experiment in self.non_zero_experiments:
                input_state = non_zero_experiment[:len(self.input_dag_nodes)]
                output_basis = non_zero_experiment[len(self.input_dag_nodes):]
                circ = self.generate_subexperiment(input_state, output_basis)
                ret[input_state + output_basis] = circ
            return ret
        else:
            return self.generate_all_experiments_naive()
    
    def run_all_experiments(self, all_experiments, backend, shots) -> dict[str, dict[str, int]]:
        ret = {
            key: backend.run(circ, shots=shots).result().get_counts()
            for key, circ in all_experiments.items()
        }
        return ret
    
    def calculate_expval(self, output_state, result_counts) -> float:
        ret = 0
        len_prefix = len(self.closed_dag_out_nodes)
        for key, value in result_counts.items():
            eigenvalue = 1
            for i, symbol in enumerate(output_state):
                if symbol != 'I' and key[-i-1] == '1':
                    eigenvalue *= -1
            for i in range(len_prefix):
                if self.closed_dag_out_nodes_Pauli[self.closed_dag_out_nodes[-i-1]] != 'I' and key[i] == '1':
                    eigenvalue *= -1
            ret += eigenvalue * value
        return ret
    
    def sparsity(self, arr, tol=1e-6):
        close_to_zero = np.abs(arr) < tol
        # Calculate the percentage
        sp = 1 - np.sum(close_to_zero) / arr.size
        return sp
    
    def num_experiments(self, array=None, ind=None) -> int:
        if array is None or self.sparsity(array) == 1:
            self.non_zero_experiments = None
            self.final_tensor_non_zero_onehot = None
            return (4 ** len(self.input_dag_nodes)) * (3 ** len(self.output_dag_nodes))
        else:
            assert ind != None
            permute = [-1 for _ in range(len(self.input_dag_nodes) + len(self.output_dag_nodes))]
            for i, var_name in enumerate(ind):
                if var_name in self.input_dag_nodes_index.keys():
                    permute[self.input_dag_nodes_index[var_name]] = i
                elif var_name in self.output_dag_nodes_index.keys():
                    permute[self.output_dag_nodes_index[var_name] + len(self.input_dag_nodes)] = i
                else:
                    raise ValueError(f'Invalid var_name {var_name}')
            new_array = array.transpose(permute).reshape(-1)
            input_states = self.generate_states(['I', 'X', 'Y', 'Z'], len(self.input_dag_nodes))
            output_states = self.generate_states(['I', 'X', 'Y', 'Z'], len(self.output_dag_nodes))
            tmp = 0
            models = []
            self.final_tensor_non_zero_onehot = []
            for input_state in input_states:
                for output_state in output_states:
                    if np.abs(new_array[tmp]) > 1e-6:
                        models.append(input_state + output_state)
                        self.final_tensor_non_zero_onehot.append(1)
                    else:
                        self.final_tensor_non_zero_onehot.append(0)
                    tmp += 1
            assert tmp == len(new_array)
            
            self.final_tensor_non_zero_onehot = np.array(self.final_tensor_non_zero_onehot)
            
            for i in range(len(self.input_dag_nodes)):
                new_models = []
                for model in models:
                    new_models.append(model[:i] + '0' + model[i+1:])
                    new_models.append(model[:i] + '1' + model[i+1:])
                    if model[i] == 'I' or model[i] == 'Z':
                        pass
                    elif model[i] == 'X':
                        new_models.append(model[:i] + '+' + model[i+1:])
                    elif model[i] == 'Y':
                        new_models.append(model[:i] + 'i' + model[i+1:])
                    else:
                        raise ValueError(f'Invalid model {model} on index {i}, not in I, X, Y, Z')
                models = list(set(new_models))
            
            new_models = []
            for model in models:
                new_models.append(model.replace('I', 'Z'))
            models = list(set(new_models))
            self.non_zero_experiments = models
            print(f'Total number of experiments: {len(self.non_zero_experiments)} / {(4 ** len(self.input_dag_nodes)) * (3 ** len(self.output_dag_nodes))}')
            return len(self.non_zero_experiments)

    
    def evaluate_by_sampling(self, backend, shots=2**12):
        all_experiments = self.generate_all_experiments()
        all_result_counts = self.run_all_experiments(all_experiments, backend, shots)
        input_states = self.generate_states(['0', '1', '+', 'i'], len(self.input_dag_nodes))
        output_states = self.generate_states(['I', 'X', 'Y', 'Z'], len(self.output_dag_nodes))

        factor = []
        for input_state in input_states:
            for output_state in output_states:
                key = (input_state + output_state).replace('I', 'Z')
                if key in all_result_counts.keys():
                    result_counts: dict[str, int] = all_result_counts[key]
                    factor.append(self.calculate_expval(output_state, result_counts) / shots\
                        / (2 ** (len(self.output_dag_nodes))))
                else:
                    factor.append(0)
        factor = np.array(factor)
        
        total = len(self.input_dag_nodes) + len(self.output_dag_nodes)
        # transform the input states from ['0', '1', '+', 'i'] to ['I', 'X', 'Y', 'Z']
        cnt = 0
        for i in range(len(self.input_dag_nodes)):
            factor = factor.reshape(4 ** cnt, 4, 4 ** (total - cnt - 1))
            # I <= |0><0|+|1><1|
            # X <= 2|+><+|-I
            # Y <= 2|i><i|-I
            # Z <= |0><0|-|1><1|
            factor[:, 0, :], factor[:, 1, :], factor[:, 2, :], factor[:, 3, :] = \
                factor[:, 0, :] + factor[:, 1, :],\
                factor[:, 2, :] * 2 - factor[:, 0, :] - factor[:, 1, :],\
                factor[:, 3, :] * 2 - factor[:, 0, :] - factor[:, 1, :],\
                factor[:, 0, :] - factor[:, 1, :]
            cnt = cnt + 1
        
        if self.final_tensor_non_zero_onehot is not None:
            factor = factor.reshape(-1)*(self.final_tensor_non_zero_onehot)
        factor = factor.reshape([4 for _ in range(total)])
        return factor, [node.op.label for node in self.input_dag_nodes] + [node.op.label for node in self.output_dag_nodes]
    
    def generate_subexperiment(self, input_state, output_basis) -> QuantumCircuit:
        dag = self.dag
        name = dag.name or None

        circuit = QuantumCircuit(
            dag.qubits,
            ClassicalRegister(len(self.output_dag_nodes) + len(self.closed_dag_out_nodes)),
            *dag.qregs.values(),
            *dag.cregs.values(),
            name=name,
            global_phase=dag.global_phase,
        )
        circuit.metadata = dag.metadata
        circuit.calibrations = dag.calibrations
        
        copy_operations = True
        for node in dag.topological_op_nodes():
            if node.op.name == 'qpd_1q':
                if self.is_this_qpd_1q_an_input_edge[node]:
                    index = self.input_dag_nodes_index[node.op.label]
                    symbol = input_state[index]
                    circuit.reset(node.qargs[0])
                    if symbol == '0':
                        pass
                    elif symbol == '1':
                        circuit.x(node.qargs[0])
                    elif symbol == '+':
                        circuit.h(node.qargs[0])
                    elif symbol == 'i':
                        circuit.h(node.qargs[0])
                        circuit.s(node.qargs[0])
                    else:
                        raise ValueError(f'Invalid symbol {symbol}')
                else:
                    index = self.output_dag_nodes_index[node.op.label]
                    symbol = output_basis[index]
                    if symbol == 'X':
                        circuit.h(node.qargs[0])
                    elif symbol == 'Y':
                        circuit.sdg(node.qargs[0])
                        circuit.h(node.qargs[0])
                    elif symbol == 'Z':
                        pass
                    else:
                        raise ValueError(f'Invalid symbol {symbol}')
                    circuit.measure(node.qargs[0], index)
            else:
                op = node.op
                if copy_operations:
                    op = copy.deepcopy(op)
                circuit._append(CircuitInstruction(op, node.qargs, node.cargs))
        for i, node in enumerate(self.closed_dag_out_nodes):
            if self.closed_dag_out_nodes_Pauli[node] == 'X':
                circuit.h(node.wire)
            elif self.closed_dag_out_nodes_Pauli[node] == 'Y':
                circuit.sdg(node.wire)
                circuit.h(node.wire)
            else:
                assert self.closed_dag_out_nodes_Pauli[node] == 'Z' or self.closed_dag_out_nodes_Pauli[node] == 'I'
            circuit.measure(node.wire, i + len(self.output_dag_nodes))
        return circuit

    def evaluate_by_classical_shadows(self, backend, num_classical_shadows_per_input_state=2**12):
        input_states = self.generate_states(['0', '1', '+', 'i'], len(self.input_dag_nodes))
        output_bases = self.generate_states(['X', 'Y', 'Z'], len(self.output_dag_nodes))
        output_states = self.generate_states(['I', 'X', 'Y', 'Z'], len(self.output_dag_nodes))

        factor = []
        for input_state in input_states:
            estimated_rhos = []
            for _ in range(num_classical_shadows_per_input_state):
                output_basis = np.random.choice(output_bases)
                circ = self.generate_subexperiment(input_state, output_basis)
                
                bitstring = None
                # Perform a single-shot experiment on this circuit
                counts_dict = backend.run(circ, shots=1).result().get_counts()
                # Get the bitstring
                for key, value in counts_dict.items():
                    assert value == 1
                    bitstring = key
                assert bitstring != None
                # Transform the bitstring to the output state, i.e. in classical shadows, Udag|b>.
                rho = 1
                for output_basis_letter, bit in zip(output_basis, bitstring[::-1][:len(self.output_dag_nodes)]):
                    # claculate the Udag|b><b|U first, then do the inverse of measurement:
                    # InvM(X)=3X-trace(X)I for a single qubit
                    # InvM(X\otimes Y)=InvM(X)\otimes InvM(Y)
                    if output_basis_letter == 'X':
                        if bit == '0': # |+><+|
                            density_this_qubit = 3 * .5 * np.array([[1, 1], [1, 1]]) - np.eye(2)
                        else: # |-><-|
                            density_this_qubit = 3 * .5 * np.array([[1, -1], [-1, 1]]) - np.eye(2)
                    elif output_basis_letter == 'Y':
                        if bit == '0': # |i><i|
                            density_this_qubit = 3 * .5 * np.array([[1, -1j], [1j, 1]]) - np.eye(2)
                        else: # |-i><-i|
                            density_this_qubit = 3 * .5 * np.array([[1, 1j], [-1j, 1]]) - np.eye(2)
                    elif output_basis_letter == 'Z':
                        if bit == '0': # |0><0|
                            density_this_qubit = 3 * np.array([[1, 0], [0, 0]]) - np.eye(2)
                        else: # |1><1|
                            density_this_qubit = 3 * np.array([[0, 0], [0, 1]]) - np.eye(2)
                    else:
                        raise ValueError(f'Invalid output_basis_letter {output_basis_letter}')
                    rho = np.kron(rho, density_this_qubit)
                coefficient = 1
                for bit in bitstring[:len(self.closed_dag_out_nodes)]:
                    if bit == '0':
                        coefficient = coefficient * 1
                    else:
                        coefficient = coefficient * -1
                estimated_rhos.append(rho * coefficient)
            
            hat_rho = np.average(estimated_rhos, axis=0)
            for output_state in output_states:
                # calculate trace(hat_rho * output_state)
                ps = 1
                for output_letter in output_state:
                    if output_letter == 'I':
                        ps = np.kron(ps, I)
                    elif output_letter == 'X':
                        ps = np.kron(ps, X)
                    elif output_letter == 'Y':
                        ps = np.kron(ps, Y)
                    elif output_letter == 'Z':
                        ps = np.kron(ps, Z)
                    else:
                        raise ValueError(f'Invalid output_letter {output_letter}')
                if len(output_state) == 0:
                    factor.append(hat_rho * ps / (2 ** (len(self.closed_dag_out_nodes) + len(self.output_dag_nodes))))
                else:
                    factor.append(np.trace(hat_rho @ ps) / (2 ** (len(self.closed_dag_out_nodes) + len(self.output_dag_nodes))))

        factor = np.array(factor)
        
        total = len(self.input_dag_nodes) + len(self.output_dag_nodes)
        # transform the input states from ['0', '1', '+', 'i'] to ['I', 'X', 'Y', 'Z']
        cnt = 0
        for i in range(len(self.input_dag_nodes)):
            factor = factor.reshape(4 ** cnt, 4, 4 ** (total - cnt - 1))
            # I <= |0><0|+|1><1|
            # X <= 2|+><+|-I
            # Y <= 2|i><i|-I
            # Z <= |0><0|-|1><1|
            factor[:, 0, :], factor[:, 1, :], factor[:, 2, :], factor[:, 3, :] = \
                factor[:, 0, :] + factor[:, 1, :],\
                factor[:, 2, :] * 2 - factor[:, 0, :] - factor[:, 1, :],\
                factor[:, 3, :] * 2 - factor[:, 0, :] - factor[:, 1, :],\
                factor[:, 0, :] - factor[:, 1, :]
            cnt = cnt + 1
        
        factor = factor.reshape([4 for _ in range(total)])
        return factor, [node.op.label for node in self.input_dag_nodes] + [node.op.label for node in self.output_dag_nodes]
    
