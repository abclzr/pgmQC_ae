from qiskit.circuit import QuantumCircuit, CircuitInstruction
import matplotlib.pyplot as plt
import copy
import time

from tqdm import tqdm
import pdb
import numpy as np
from qiskit_addon_cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)
from qiskit_addon_cutting import cut_wires, expand_observables
from qiskit_addon_cutting import partition_problem
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_cutting import reconstruct_expectation_values
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_cutting import generate_cutting_experiments
from qiskit_ibm_runtime import SamplerV2, Batch
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

def build_circuit_from_subdag(dag, enable, copy_operations=True):
    name = dag.name or None
    circuit = QuantumCircuit(
        dag.qubits,
        dag.clbits,
        *dag.qregs.values(),
        *dag.cregs.values(),
        name=name,
        global_phase=dag.global_phase,
    )
    circuit.metadata = dag.metadata
    circuit.calibrations = dag.calibrations
    
    for node in dag.topological_op_nodes():
        if enable[node]:
            op = node.op
            if copy_operations:
                op = copy.deepcopy(op)
            circuit._append(CircuitInstruction(op, node.qargs, node.cargs))
    return circuit


def plot_ps(ps, filename):
    # Plot
    plt.close()
    x = np.arange(len(ps))

    plt.bar(x, ps)

    # Adding labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

def qiskit_wire_cut(circuit, observable, partition_label, qubits_limit, seed=111):
    qc_1 = circuit
    partitioned_problem = partition_problem(
        circuit=qc_1, partition_labels=partition_label, observables=observable.paulis
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    bases = partitioned_problem.bases
    return subcircuits, subobservables, bases

def qiskit_auto_wire_cut(circuit, observable, qubits_limit, seed=111):
    # Specify settings for the cut-finding optimizer
    optimization_settings = OptimizationParameters(seed=111)
    optimization_settings.gate_lo = False

    # Specify the size of the QPUs available
    device_constraints = DeviceConstraints(qubits_per_subcircuit=qubits_limit)

    cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
    print(
        f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
        f'overhead of {metadata["sampling_overhead"]}.\n'
        f'Lowest cost solution found: {metadata["minimum_reached"]}.'
    )
    for cut in metadata["cuts"]:
        print(f"{cut[0]} at circuit instruction index {cut[1]}")
    cut_circuit.draw("mpl", scale=0.8, fold=-1)
    qc_w_ancilla = cut_wires(cut_circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)
    qc_w_ancilla.draw("mpl", scale=0.8, fold=-1)
    partitioned_problem = partition_problem(
    circuit=qc_w_ancilla, observables=observables_expanded
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    print(
        f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}"
    )
    return subcircuits, subobservables

def qiskit_reconstruction(backend, subcircuits, subobservables, observable, total_shots, num_trials=1):
    print('Generating cutting experiments...')
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=subcircuits, observables=subobservables, num_samples=np.inf
    )
    
    print('Transpiling subexperiments...')
    # Transpile the subexperiments to ISA circuits
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    isa_subexperiments = {
        label: pass_manager.run(partition_subexpts)
        for label, partition_subexpts in tqdm(subexperiments.items())
    }
    
    print('Calculating total number of experiments and shots...')
    total_experiments = 0
    for label, subsystem_subexpts in isa_subexperiments.items():
        total_experiments += len(subsystem_subexpts)
    shots_per_experiment = total_shots // total_experiments
    print(f'Total experiments: {total_experiments}')
    print(f'Total shots: {total_shots}')
    print(f'Shots per experiment: {shots_per_experiment}')
    
    expval_list = []
    execution_time_ms_list = []
    for i in tqdm(range(num_trials)):
        # Submit each partition's subexperiments to the Qiskit Runtime Sampler
        # primitive, in a single batch so that the jobs will run back-to-back.
        with Batch(backend=backend) as batch:
            sampler = SamplerV2(mode=batch)
            jobs = {
                label: sampler.run(subsystem_subexpts, shots=shots_per_experiment)
                for label, subsystem_subexpts in isa_subexperiments.items()
            }
        # Retrieve results
        results = {label: job.result() for label, job in jobs.items()}
        start_time = time.time()  # Record the start time in seconds
        reconstructed_expval_terms = reconstruct_expectation_values(
            results,
            coefficients,
            subobservables,
        )
        reconstructed_expval = np.dot(reconstructed_expval_terms, observable.coeffs)
        end_time = time.time()  # Record the end time

        execution_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

        expval_list.append(reconstructed_expval)
        execution_time_ms = (end_time - start_time) * 1000
        execution_time_ms_list.append(execution_time_ms)
        # print(f"Reconstructed expectation value: {np.real(np.round(reconstructed_expval, 8))}")
    return expval_list, {
        'qiskit_total_experiments': total_experiments,
        'qiskit_total_shots': total_shots,
        'qiskit_shots_per_experiment': shots_per_experiment,
        'qiskit_execution_time_ms_list': execution_time_ms_list
    }