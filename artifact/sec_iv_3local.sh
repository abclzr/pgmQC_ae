#!/bin/bash
# This script test the circuit cutting recombination for 3-local HWEA.
python generate_tensors.py --task_name 3local --n_qubits 8 --d 1 --qubit_constrain 5 --total_shots 320000 --num_trials 100
python generate_tensors.py --task_name 3local --n_qubits 8 --d 1 --qubit_constrain 5 --total_shots 640000 --num_trials 100
