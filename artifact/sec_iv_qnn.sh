#!/bin/bash
# This script test the circuit cutting recombination for an end-to-end QNN example. Takes approximately 12 hours to run.
python generate_tensors.py --task_name QNN --n_qubits 4 --d 2 --qubit_constrain 3 --total_shots 320000 --num_trials 10
# python generate_tensors.py --task_name QNN --n_qubits 4 --d 2 --qubit_constrain 3 --total_shots 640000 --num_trials 10