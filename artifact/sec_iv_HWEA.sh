#!/bin/bash
# This script test the circuit cutting recombination for HWEA_RYRZ_CZ with different number of layers.
python generate_tensors.py --task_name HWEA_RYRZ_CZ --n_qubits 8 --d 1 --qubit_constrain 5 --total_shots 10000 --num_trials 1
python generate_tensors.py --task_name HWEA_RYRZ_CZ --n_qubits 8 --d 1 --qubit_constrain 5 --total_shots 20000 --num_trials 1
python generate_tensors.py --task_name HWEA_RYRZ_CZ --n_qubits 8 --d 2 --qubit_constrain 5 --total_shots 10000 --num_trials 1
python generate_tensors.py --task_name HWEA_RYRZ_CZ --n_qubits 8 --d 2 --qubit_constrain 5 --total_shots 20000 --num_trials 1
python generate_tensors.py --task_name HWEA_RYRZ_CZ --n_qubits 8 --d 4 --qubit_constrain 5 --total_shots 10000 --num_trials 1
python generate_tensors.py --task_name HWEA_RYRZ_CZ --n_qubits 8 --d 4 --qubit_constrain 5 --total_shots 20000 --num_trials 1