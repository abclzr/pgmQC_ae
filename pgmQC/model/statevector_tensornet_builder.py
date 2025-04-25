import numpy as np
import pdb

from qiskit import dagcircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from qiskit.quantum_info import Statevector
from qiskit.dagcircuit.dagcircuit import DAGCircuit

class StatevectorTensorNetworkBuilder:
    def __init__(self, dag: DAGCircuit, noise_inject=False):
        self.dag = dag
        self.dag.draw(filename="dag.png")
        self.basis_lookup = {'|0>': 0, '|1>': 1}
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
        basis_lookup = self.basis_lookup
        
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
                factor = DiscreteFactor([var_name_lookup[connected_vars[0]]],
                        cardinality=[2],
                        values=[1., 0.],
                        state_names={var_name_lookup[var] : ['|0>', '|1>'] for var in connected_vars}
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
                basis = [ Statevector([1, 0]), Statevector([0, 1]) ]
                initial_basis = basis

                n = opnode.op.num_qubits
                while n > 1:
                    new_initial_basis = []
                    for a in initial_basis:
                        for b in basis:
                            new_initial_basis.append(a.expand(b))
                    initial_basis = new_initial_basis
                    n = n-1

                # for every input variables state, get the output variables state
                values = []
                for basis in initial_basis:
                    output_statevector = basis.evolve(opnode.op)                    
                    values.append(output_statevector.data)

                # create the factor
                values = np.concatenate(values, axis=0)
                factor = DiscreteFactor([var_name_lookup[var] for var in connected_vars],
                        cardinality=[2 for var in connected_vars],
                        values=values,
                        state_names={var_name_lookup[var] : ['|0>', '|1>'] for var in connected_vars}
                )
                factors.add(factor)

            elif isinstance(opnode, dagcircuit.DAGOutNode):
                pass
            else:
                raise TypeError(opnode)

        for var in vars:
            if enable == None or enable[var[0]]:
                markov_net.add_node(var_name_lookup[var])
            elif enable[var[1]]:
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
        nids = []
        for node, count in nodes_count.items():
            if count == 1:
                uncontracted_nodes.append(node)
                for var in vars:
                    if self.var_name_lookup[var] == node:
                        nids.append(var[2]._index)
            else:
                assert count == 0

        print(nids)
        
        return markov_net, uncontracted_nodes
