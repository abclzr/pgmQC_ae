from time import time
from statistics import mean, stdev
import numpy as np
np.set_printoptions(suppress=True)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random.utils import random_circuit
from qiskit.dagcircuit.dagnode import DAGOutNode
from qiskit.quantum_info import SparsePauliOp, DensityMatrix, random_clifford
from qiskit_aer import AerSimulator

from qc_bn import qc_bn
from recurse import recurse

def evaluate (circuit: QuantumCircuit):

    print(circuit)

    dag, bn, query_vars = qc_bn (circuit)

    cut_sizes, factor, models = recurse (dag, bn, query_vars, 0)

    tuples = []
    for key in models:
        pauli_string = ["I"] * len(query_vars)
        for tup in key:
            assert isinstance(tup[0][1],DAGOutNode)
            if  pauli_string[tup[0][1].wire.index] == "X" and tup[1]=="Z":
                pauli_string[tup[0][1].wire.index] = "Y"
            elif pauli_string[tup[0][1].wire.index] == "Z" and tup[1]=="X":
                pauli_string[tup[0][1].wire.index] = "Y"
            else:
                pauli_string[tup[0][1].wire.index] = tup[1]
        pauli_string.reverse()
        tuples.append( ("".join(pauli_string), models[key]) )

    print(tuples)
    dm_ref = DensityMatrix.from_instruction(circuit)
    print(SparsePauliOp.from_operator(dm_ref))
    pgm_sparse = SparsePauliOp.from_list(tuples)
    print(pgm_sparse)
    dm_dut = DensityMatrix(pgm_sparse)
    assert(np.allclose(dm_ref,dm_dut))
    print(cut_sizes)
    quit()
    return(pgm_sparse, 0, 0)

# while True:
#     qc = QuantumCircuit(2)
#     qc.h(0)
#     qc.cx(0,1)
#     evaluate(qc)
#     quit()

qubit_counts = range(8,10)
trial_count = 32

t_pgm_times = []
t_pgm_stdevs = []
t_stabilizer_times = []
t_stabilizer_stdevs = []
t_tensor_times = []
t_tensor_stdevs = []

clifford_pgm_times = []
clifford_pgm_stdevs = []
clifford_stabilizer_times = []
clifford_stabilizer_stdevs = []
clifford_tensor_times = []
clifford_tensor_stdevs = []

for qubit_count in qubit_counts:

    print("qubit_count = ")
    print(qubit_count)

    qft_pgm_trials = []
    qft_stabilizer_trials = []
    qft_tensor_trials = []

    for _ in range(trial_count):

        from qiskit.circuit.library import QFT
        from qiskit.synthesis import synth_clifford_full

        print("synthesizing QFT circuit")

        qft_circuit = QFT(
            num_qubits=qubit_count,
            approximation_degree=0,
            do_swaps=False,
            inverse=False,
            insert_barriers=False,
            name='qft'
        ).decompose().decompose()

        qft_data, qft_evidence_time, qft_evaluation_time = evaluate(qft_circuit)

    quit()

    t_pgm_trials = []
    t_stabilizer_trials = []
    t_tensor_trials = []

    for _ in range(trial_count):

        print("synthesizing t circuit")
        t_circuit = random_circuit(
            num_qubits=qubit_count,
            depth=qubit_count,
            max_operands=1,
            measure=False,
            conditional=False,
            reset=False,
            seed=None
        )

        pgm_data, pgm_evidence_time, pgm_evaluation_time = evaluate(t_circuit)
        t_pgm_trials.append(pgm_evidence_time+pgm_evaluation_time)

        t_circuit.save_statevector()

        stabilizer_simulator = AerSimulator(method='extended_stabilizer')

        print("transpiling for stabilizer simulator")
        t_circuit = transpile(t_circuit, stabilizer_simulator)

        print("running stabilizer simulator")
        start = time()
        stabilizer_run = stabilizer_simulator.run(t_circuit)
        stabilizer_statevector = stabilizer_run.result().get_statevector()
        end = time()
        t_stabilizer_trials.append(end-start)

        print("running tensor simulator")
        tensor_network_simulator = AerSimulator(
            # method='tensor_network',
            # device='GPU'
        )
        start = time()
        tensor_network_run = tensor_network_simulator.run(t_circuit)
        tensor_statevector = tensor_network_run.result().get_statevector()
        end = time()
        t_tensor_trials.append(end-start)

        print("checking statevector equivalency")
        print("pgm_data")
        print(pgm_data)
        print(tensor_statevector)
        tensor_sparse = SparsePauliOp.from_operator(tensor_statevector)
        print("tensor_sparse")
        print(tensor_sparse)
        assert(pgm_data.equiv(tensor_sparse))
        assert(tensor_statevector.equiv(stabilizer_statevector,rtol=1e-1,atol=1e-1))

    t_pgm_times.append(mean(t_pgm_trials))
    t_pgm_stdevs.append(stdev(t_pgm_trials))
    t_stabilizer_times.append(mean(t_stabilizer_trials))
    t_stabilizer_stdevs.append(stdev(t_stabilizer_trials))
    t_tensor_times.append(mean(t_tensor_trials))
    t_tensor_stdevs.append(stdev(t_tensor_trials))

    clifford_pgm_trials = []
    clifford_stabilizer_trials = []
    clifford_tensor_trials = []

    from qiskit.synthesis import synth_clifford_full

    for _ in range(trial_count):

        print("synthesizing clifford circuit")
        clifford_circuit = synth_clifford_full(random_clifford(qubit_count))

        pgm_data, pgm_evidence_time, pgm_evaluation_time = evaluate(clifford_circuit)
        clifford_pgm_trials.append(pgm_evidence_time+pgm_evaluation_time)

        clifford_circuit.save_state()

        stabilizer_simulator = AerSimulator(method='stabilizer')

        print("running stabilizer simulator")
        start = time()
        stabilizer_run = stabilizer_simulator.run(clifford_circuit)
        stabilizer_data = stabilizer_run.result().data()
        end = time()
        clifford_stabilizer_trials.append(end-start)
        print("getting stabilizer data")

        tensor_network_simulator = AerSimulator(
            # device='GPU',
            # method='tensor_network'
        )
        print("running tensor simulator")
        start = time()
        tensor_network_run = tensor_network_simulator.run(clifford_circuit)
        tensor_data = tensor_network_run.result().data()
        end = time()
        clifford_tensor_trials.append(end-start)
        print("getting tensor data")

        print("checking data equivalency")
        print("pgm_data")
        print(pgm_data)
        # tensor_sparse = SparsePauliOp.from_operator(tensor_data['tensor_network'])
        # print("tensor_sparse")
        # print(tensor_sparse)
        # assert(pgm_data.equiv(tensor_sparse))
        print("stabilizer_data")
        print(stabilizer_data)
        # assert(tensor_data['tensor_network'].equiv(stabilizer_data))

    clifford_pgm_times.append(mean(clifford_pgm_trials))
    clifford_pgm_stdevs.append(stdev(clifford_pgm_trials))
    clifford_stabilizer_times.append(mean(clifford_stabilizer_trials))
    clifford_stabilizer_stdevs.append(stdev(clifford_stabilizer_trials))
    clifford_tensor_times.append(mean(clifford_tensor_trials))
    clifford_tensor_stdevs.append(stdev(clifford_tensor_trials))

    import matplotlib.pyplot as plt
    fig, (ax_t, ax_clifford) = plt.subplots(1, 2)

    ax_t.set_title("T circuit")
    ax_t.errorbar(range(2,qubit_count+1), t_pgm_times, t_pgm_stdevs, label='pgm')
    ax_t.errorbar(range(2,qubit_count+1), t_stabilizer_times, t_stabilizer_stdevs, label='stabilizer')
    ax_t.errorbar(range(2,qubit_count+1), t_tensor_times, t_tensor_stdevs, label='tensor')
    ax_t.legend()

    ax_clifford.set_title("Clifford circuit")
    ax_clifford.errorbar(range(2,qubit_count+1), clifford_pgm_times, clifford_pgm_stdevs, label='pgm')
    ax_clifford.errorbar(range(2,qubit_count+1), clifford_stabilizer_times, clifford_stabilizer_stdevs, label='stabilizer')
    ax_clifford.errorbar(range(2,qubit_count+1), clifford_tensor_times, clifford_tensor_stdevs, label='tensor')
    ax_clifford.legend()

    plt.savefig("figure.pdf")