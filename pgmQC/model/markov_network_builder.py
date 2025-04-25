import numpy as np
import pdb

from qiskit import dagcircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from qiskit.quantum_info import Pauli, SparsePauliOp, DensityMatrix, random_clifford
from qiskit.dagcircuit.dagcircuit import DAGCircuit

class MarkovNetworkBuilder:
    def __init__(self, dag: DAGCircuit, noise_inject=False):
        self.dag = dag
        self.dag.draw(filename="dag.png")
        self.pauli_index_lookup = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        self.nodes = list(self.dag.nodes())
        self.vars = list(self.dag.edges())
        self.var_name_lookup = {}
        for i, var in enumerate(self.vars):
            self.var_name_lookup[var] = f'v{i}'
        self.noise_inject = noise_inject
        lam = 0.3
        self.depolarizing_factor = 1-lam/2
    
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
    def build(self, enable=None):
        nodes = self.nodes
        vars = self.vars
        var_name_lookup = self.var_name_lookup
        pauli_index_lookup = self.pauli_index_lookup
        
        markov_net = MarkovNetwork()
        factors = set()
        
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
                if self.noise_inject:
                    variable_names = [var_name_lookup[var] for var in connected_vars]
                    factor = DiscreteFactor(variable_names,
                            cardinality=[4],
                            values=[.5, 0, 0, .5],
                            state_names={variable_name : ['I', 'X', 'Y', 'Z'] for variable_name in variable_names}
                    )
                    factors.add(factor)
                    for variable_name in variable_names:
                        factor = DiscreteFactor([variable_name, variable_name + '_after_noise'],
                                                cardinality=[4, 4],
                                                values=[1., 0, 0, 0,
                                                        0, self.depolarizing_factor, 0, 0,
                                                        0, 0, self.depolarizing_factor, 0,
                                                        0, 0, 0, self.depolarizing_factor],
                                                state_names={variable_name : ['I', 'X', 'Y', 'Z'],
                                                             variable_name + '_after_noise' : ['I', 'X', 'Y', 'Z']}
                        )
                        factors.add(factor)
                else:
                    factor = DiscreteFactor([var_name_lookup[connected_vars[0]]],
                            cardinality=[4],
                            values=[.5, 0, 0, .5],
                            state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                    )
                    factors.add(factor)
            elif isinstance(opnode, dagcircuit.DAGOpNode):
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
                if self.noise_inject:
                    variable_names = [var_name_lookup[var] for var in connected_vars]
                    len_vn = len(variable_names)
                    for i in range(len_vn // 2):
                        variable_names[i] = variable_names[i] + '_after_noise'
                    factor = DiscreteFactor(variable_names,
                            cardinality=[4 for var in connected_vars],
                            values=values,
                            state_names={variable_name : ['I', 'X', 'Y', 'Z'] for variable_name in variable_names}
                    )
                    factors.add(factor)
                    for variable_name in variable_names[len_vn // 2:]:
                        factor = DiscreteFactor([variable_name, variable_name + '_after_noise'],
                                                cardinality=[4, 4],
                                                values=[1., 0, 0, 0,
                                                        0, self.depolarizing_factor, 0, 0,
                                                        0, 0, self.depolarizing_factor, 0,
                                                        0, 0, 0, self.depolarizing_factor],
                                                state_names={variable_name : ['I', 'X', 'Y', 'Z'],
                                                             variable_name + '_after_noise' : ['I', 'X', 'Y', 'Z']}
                        )
                        factors.add(factor)
                else:
                    factor = DiscreteFactor([var_name_lookup[var] for var in connected_vars],
                            cardinality=[4 for var in connected_vars],
                            values=values,
                            state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                    )
                    factors.add(factor)

            elif isinstance(opnode, dagcircuit.DAGOutNode):
                connected_vars = connected(opnode)
                factor = DiscreteFactor([var_name_lookup[connected_vars[0]]],
                    cardinality=[4],
                    values=[0, 0, 0, 1.],
                    state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                )
                factors.add(factor)
            else:
                raise TypeError(opnode)

        for var in vars:
            if enable == None or enable[var[0]]:
                markov_net.add_node(var_name_lookup[var])
                if self.noise_inject:
                    markov_net.add_node(var_name_lookup[var] + '_after_noise')                    
            elif enable[var[1]]:
                if self.noise_inject:
                    markov_net.add_node(var_name_lookup[var] + '_after_noise')
                else:
                    markov_net.add_node(var_name_lookup[var])


        for factor in factors:
            markov_net.add_factors(factor)
            edges = [(factor.variables[i], factor.variables[j])
                    for i in range(len(factor.variables))
                    for j in range(i + 1, len(factor.variables))]
            markov_net.add_edges_from(ebunch=edges)
        
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

        
        return markov_net, uncontracted_nodes
