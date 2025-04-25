import qiskit.circuit.library as library
import math, qiskit, random, logging
import networkx as nx
import numpy as np
from typing import Tuple
from qiskit import QuantumCircuit

from pgmQC.test_dataset.Supremacy import Qgrid_original, Qgrid_Sycamore
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_addon_cutting.instructions import CutWire, Move

def w_state(num_qubits: int):
    """
    W state
    https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/awards/teach_me_qiskit_2018/w_state/W%20State%201%20-%20Multi-Qubit%20Systems.ipynb
    Single qubit gates are proxies only.
    Circuit structure is accurate.
    """

    def F_gate(circ, i, j, n, k):
        theta = np.arccos(np.sqrt(1 / (n - k + 1)))
        circ.ry(-theta, j)
        circ.cz(i, j)
        circ.ry(theta, j)

    circuit = qiskit.QuantumCircuit(num_qubits)
    circuit.x(num_qubits - 1)
    for qubit in range(num_qubits - 1, -1, -1):
        F_gate(circuit, qubit, qubit - 1, num_qubits, num_qubits - qubit)
    for qubit in range(num_qubits - 2, -1, -1):
        circuit.cx(qubit, qubit + 1)
    return circuit


def ghz(num_qubits: int):
    """
    GHZ state
    https://qiskit.org/documentation/stable/0.24/tutorials/noise/9_entanglement_verification.html
    """
    circuit = qiskit.QuantumCircuit(num_qubits)
    circuit.h(0)
    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


def gen_supremacy(
    height,
    width,
    depth,
    order=None,
    singlegates=True,
    mirror=False,
    barriers=False,
    measure=False,
    regname=None,
):
    """
    Calling this function will create and return a quantum supremacy
    circuit based on the implementations in
    https://www.nature.com/articles/s41567-018-0124-x and
    https://github.com/sboixo/GRCS.
    """

    grid = Qgrid_original.Qgrid(
        height,
        width,
        depth,
        order=order,
        mirror=mirror,
        singlegates=singlegates,
        barriers=barriers,
        measure=measure,
        regname=regname,
    )

    circ = grid.gen_circuit()

    return circ

def gen_sycamore(
    height,
    width,
    depth,
    order=None,
    singlegates=True,
    barriers=False,
    measure=False,
    regname=None,
):
    """
    Calling this function will create and return a quantum supremacy
    circuit as found in https://www.nature.com/articles/s41586-019-1666-5
    """

    grid = Qgrid_Sycamore.Qgrid(
        height,
        width,
        depth,
        order=order,
        singlegates=singlegates,
        barriers=barriers,
        measure=measure,
        regname=regname,
    )

    circ = grid.gen_circuit()

    return circ


def construct_qaoa_plus(P, G, params, reg_name):
    assert len(params) == 2 * P, "Number of parameters should be 2P"

    nq = len(G.nodes())
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(nq, reg_name))

    # Initial state
    circ.h(range(nq))

    gammas = [param for i, param in enumerate(params) if i % 2 == 0]
    betas = [param for i, param in enumerate(params) if i % 2 == 1]
    for i in range(P):
        # Phase Separator Unitary
        for edge in G.edges():
            q_i, q_j = edge
            circ.rz(gammas[i] / 2, [q_i, q_j])
            circ.cx(q_i, q_j)
            circ.rz(-1 * gammas[i] / 2, q_j)
            circ.cx(q_i, q_j)

        # Mixing Unitary
        for q_i in range(nq):
            circ.rx(-2 * betas[i], q_i)

    return circ

# This HWEA comes from the paper:
# https://www.researchgate.net/publication/362489244_Finite-size_criticality_in_fully_connected_spin_models_on_superconducting_quantum_hardware
def get_HWEA_Michele_Grossi(n_qubits, d=1, qubit_constrain=4, pruning_ratio=0.):
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
            if np.random.rand() < pruning_ratio:
                continue
            a = np.random.uniform(-np.pi, np.pi)
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
        if np.random.rand() < pruning_ratio:
            continue
        a = np.random.uniform(-np.pi, np.pi)
        circuit.ry(a, j)
        circuit_for_qiskit_cut.ry(a, map_to_physical_qubits[j])
        ob_str[map_to_physical_qubits[j]] = "Z"
    observable = SparsePauliOp(["".join(['Z'] * n_qubits)])
    observable_expanded = SparsePauliOp(["".join(reversed(ob_str))])
    
    return circuit, circuit_for_qiskit_cut, observable, observable_expanded, partition_label

