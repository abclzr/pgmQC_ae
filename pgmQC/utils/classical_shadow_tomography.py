import numpy as np
import matplotlib.pyplot as plt
import qiskit
import pdb
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import fake_backend

def channel(N,qc):
    '''create an N qubit GHZ state '''
    qc.h(0)
    if N>=2: qc.cx(0,1)
    if N>=3: qc.cx(0,2)
    if N>=4: qc.cx(1,3)
    if N>4: raise NotImplementedError(f"{N} not implemented!")

def bitGateMap(qc,g,qi):
    '''Map X/Y/Z string to qiskit ops'''
    if g=="X":
        qc.h(qi)
    elif g=="Y":
        qc.sdg(qi)
        qc.h(qi)
    elif g=="Z":
        pass
    else:
        raise NotImplementedError(f"Unknown gate {g}")

def Minv(N,X):
    '''inverse shadow channel'''
    return ((2**N+1.))*X - np.eye(2**N)

def classical_shadow_tomography(qc : qiskit.QuantumCircuit, backend : fake_backend.FakeBackendV2, nShadows = 100):
    sampler = SamplerV2(backend)

    pauli_list = [
        np.eye(2),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0, -1.0j], [1.0j, 0.0]]),
        np.array([[1.0, 0.0], [0.0, -1.0]]),
    ]
    s_to_pauli = {
        "I": pauli_list[0],
        "X": pauli_list[1],
        "Y": pauli_list[2],
        "Z": pauli_list[3],
    }
    reps = 1
    N = qc.num_qubits
    rng = np.random.default_rng(1717)
    cliffords = [qiskit.quantum_info.random_clifford(N, seed=rng) for _ in range(nShadows)]
    print(f"length of cliffords: , {len(cliffords)}")
    results = []
    for cliff in cliffords:
        qc_c  = qc.compose(cliff.to_circuit())
        qc_c.measure_all()
        qc_c = transpile(qc_c, backend)
        job = sampler.run([qc_c], shots=reps)
        counts = job.result()[0].data.meas.get_counts()
        # counts = qiskit.quantum_info.Statevector(qc_c).sample_counts(reps)
        results.append(counts)
    
    
    shadows = []
    for cliff, res in zip(cliffords, results):
        mat = cliff.adjoint().to_matrix()
        for bit,count in res.items():
            Ub = mat[:,int(bit,2)] # this is Udag|b>
            shadows.append(Minv(N,np.outer(Ub,Ub.conj()))*count)

    rho_shadow = np.sum(shadows,axis=0)/(nShadows*reps)
    return rho_shadow