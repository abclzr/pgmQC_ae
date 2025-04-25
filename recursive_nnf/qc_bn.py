from qiskit import QuantumCircuit, dagcircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli, SparsePauliOp, DensityMatrix

import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD

def qc_bn (circuit: QuantumCircuit):

    print("building bayesian network")

    dag = circuit_to_dag(circuit)
    mdg = nx.MultiDiGraph(dag.edges())
    bn = BayesianNetwork(nx.line_graph(mdg))

    query_vars = []
    for node in bn:
        
        if ( isinstance(node[0], dagcircuit.DAGInNode) ):
            value_0 = 0.5
            value_1 = 0.5
            cpd = TabularCPD (
                variable = node,
                variable_card = 4,
                values = [[value_0], [0], [0], [value_1]],
                state_names={node: ['I', 'X', 'Y', 'Z']}
            )
            bn.add_cpds(cpd)

        elif ( isinstance(node[0], dagcircuit.DAGOpNode) ):

            paulis = [ Pauli('I'), Pauli('X'), Pauli('Y'), Pauli('Z') ]
            strings = paulis

            n = node[0].op.num_qubits
            while n>1:
                new_strings = []
                for a in strings:
                    for b in paulis:
                        if node[0].qargs[0].index < node[0].qargs[1].index:
                            new_strings.append(a.tensor(b))
                        else:
                            new_strings.append(a.expand(b))
                strings = new_strings
                n = n-1

            values = { 'I':[], 'X':[], 'Y':[], 'Z':[] }
            for string in strings:

                for pauli in values:
                    values[pauli].append(0)

                sparse = SparsePauliOp.from_operator(DensityMatrix(string).evolve(node[0].op))
                for label,coeff in sparse.label_iter():
                    # values[label[node[0].op.num_qubits-node[0].qargs.index(node[2])-1]][-1] = pow(coeff,1/node[0].op.num_qubits)
                    if node[0].op.num_qubits-node[0].qargs.index(node[2])-1 == 0:
                        pass
                    else:
                        coeff = abs(coeff)
                    if abs(coeff-1.0)<1e-8:
                        coeff = 1.0
                    values[label[node[0].op.num_qubits-node[0].qargs.index(node[2])-1]][-1] = coeff

            values = [ values['I'], values['X'], values['Y'], values['Z'] ]

            evidence = list(bn.predecessors(node))
            if len(evidence)>1:
                if evidence[0][2].index < evidence[1][2].index:
                    evidence.reverse()

            if len(evidence)>1:
                state_names = {
                    list(bn.predecessors(node))[0]: ['I', 'X', 'Y', 'Z'],
                    list(bn.predecessors(node))[1]: ['I', 'X', 'Y', 'Z'],
                    node: ['I', 'X', 'Y', 'Z']
                }
            else:
                state_names = {
                    list(bn.predecessors(node))[0]: ['I', 'X', 'Y', 'Z'],
                    node: ['I', 'X', 'Y', 'Z']
                }
            cpd = TabularCPD (
                variable = node,
                variable_card = 4,
                values = values,
                evidence = evidence,
                evidence_card = node[0].op.num_qubits * [4],
                state_names=state_names,
            )
            bn.add_cpds(cpd)

        else:
            quit()

        if isinstance ( node[1], dagcircuit.DAGOutNode ):
            query_vars.append(node)

    return dag, bn, query_vars