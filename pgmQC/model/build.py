from copy import deepcopy
import pdb
from qiskit import QuantumCircuit, dagcircuit, transpile
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.quantum_info import Pauli, SparsePauliOp, DensityMatrix, random_clifford
from qiskit.circuit.random.utils import random_circuit
import numpy as np
np.set_printoptions(suppress=True)
# BASE CASE: EVALUATE VIA BAYESIAN NETWORK
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagnode import DAGOutNode
from qiskit.dagcircuit.dagcircuit import DAGCircuit
import networkx as nx

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD

from pgmQC.model.markov_network_builder import MarkovNetworkBuilder

def build_pgm(circuit: QuantumCircuit):

    # BASE CASE: EVALUATE VIA QUANTUM CIRCUIT
    print(circuit)

    print("building bayesian network")
    dag = circuit_to_dag(circuit)
    dag.draw(filename="dag.png")
    mdg = nx.MultiDiGraph(dag.edges())

    bn = BayesianNetwork(nx.line_graph(mdg))
    bn_cpds = set()
    bn_ac_cpds = set()

    outputs = set()
    weight_lookup = {}
    weight_activation = {}
    for node in bn:
        from random import randint
        if ( isinstance(node[0], dagcircuit.DAGInNode) ):
            value_0 = 0.5
            value_1 = 0.5
            cpd = TabularCPD (
                variable = node,
                variable_card = 4,
                values = [[value_0], [0], [0], [value_1]],
                state_names={node: ['I', 'X', 'Y', 'Z']}
            )
            bn_cpds.add(cpd)

            id_0 = randint(256,1048576)
            weight_lookup[id_0] = value_0
            weight_activation[id_0] = node
            id_1 = randint(256,1048576)
            weight_lookup[id_1] = value_1
            weight_activation[id_1] = node
            cpd = TabularCPD (
                variable = node,
                variable_card = 4,
                values = [[id_0], [0], [0], [id_1]],
                state_names={node: ['I', 'X', 'Y', 'Z']}
            )
            bn_ac_cpds.add(cpd)

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
            values_ac = { 'I':[], 'X':[], 'Y':[], 'Z':[] }
            for string in strings:

                for pauli in values:
                    values[pauli].append(0)
                    values_ac[pauli].append(0)

                sparse = SparsePauliOp.from_operator(DensityMatrix(string).evolve(node[0].op))
                for label,coeff in sparse.label_iter():
                    # values[label[node[0].op.num_qubits-node[0].qargs.index(node[2])-1]][-1] = pow(coeff,1/node[0].op.num_qubits)
                    if node[0].op.num_qubits-node[0].qargs.index(node[2])-1 == 0: # this is the target qubit
                        pass
                    else: # this is the control qubit
                        coeff = abs(coeff)
                    if abs(coeff-1.0)<1e-8:
                        coeff = 1.0
                    values[label[node[0].op.num_qubits-node[0].qargs.index(node[2])-1]][-1] = coeff
                    if coeff != 0.0 or coeff != 1.0:
                        identifier = randint(256,1048576)
                        weight_lookup[identifier] = coeff
                        weight_activation[identifier] = node
                        values_ac[label[node[0].op.num_qubits-node[0].qargs.index(node[2])-1]][-1] = identifier
                    else:
                        values_ac[label[node[0].op.num_qubits-node[0].qargs.index(node[2])-1]][-1] = coeffs

            values = [ values['I'], values['X'], values['Y'], values['Z'] ]
            values_ac = [ values_ac['I'], values_ac['X'], values_ac['Y'], values_ac['Z'] ]

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
            bn_cpds.add(cpd)

            cpd = TabularCPD (
                variable = node,
                variable_card = 4,
                values = values_ac,
                evidence = evidence,
                evidence_card = node[0].op.num_qubits * [4],
                state_names=state_names,
            )
            bn_ac_cpds.add(cpd)

        else:
            quit()

        if isinstance ( node[1], dagcircuit.DAGOutNode ):
            outputs.add(node)

    for cpd in bn_cpds:
        bn.add_cpds(cpd)
    return bn


def build_markov_networks(circuit: QuantumCircuit, dag: DAGCircuit):
    markov_net_builder = MarkovNetworkBuilder(dag)
    mn = markov_net_builder.build()
    return mn

