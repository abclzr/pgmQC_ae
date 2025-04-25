import numpy as np
from pgmQC.utils.setting import *

index_to_pauli = {0: I, 1: X, 2: Y, 3: Z}

def decompose_index_4base(index, n_qubits):
    ret = np.zeros(n_qubits)
    for i in range(n_qubits-1, -1, -1):
        ret[i] = index % 4
        index = index // 4
    return ret

def density_to_paulistring(density, n_qubits):
    assert density.shape == (np.power(2, n_qubits), np.power(2, n_qubits))
    l = []
    for i in range(np.power(4, n_qubits)):
        di = decompose_index_4base(i, n_qubits)
        base = 1
        for j in range(n_qubits):
            base = np.kron(base, index_to_pauli[di[j]])
        base = base.reshape(-1) # type: ignore
        l.append(base)
    l = np.array(l).reshape(np.power(4, n_qubits), np.power(4, n_qubits)).transpose()
    # l * x = density => x = inv(l) * density
    invl = np.linalg.inv(l)
    density = density.reshape(np.power(4, n_qubits), 1)
    x = np.matmul(invl, density).reshape(np.power(4, n_qubits))
    return x.real

def paulistring_to_density(pauliString, n_qubits):
    pauliString = pauliString.flatten()
    assert pauliString.shape == (np.power(4, n_qubits))
    l = []
    for i in range(np.power(4, n_qubits)):
        di = decompose_index_4base(i, n_qubits)
        base = 1
        for j in range(n_qubits):
            base = np.kron(base, index_to_pauli[di[j]])
        l.append(base)

    density = np.zeros(np.power(2, n_qubits), np.power(2, n_qubits))
    for blochfactor, base in zip(pauliString, l):
        density = density + blochfactor * base
    return density