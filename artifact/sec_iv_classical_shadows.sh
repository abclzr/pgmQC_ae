#!/bin/bash
# This script test the classical shadows. Takes approximately 2 hours to run.
python generate_tensors.py --task_name HWEA_RYRZ_CZ --n_qubits 8 --d 2 --qubit_constrain 5 --total_shots 10000 --num_trials 1 --with_classical_shadows
python generate_tensors.py --task_name 3local --n_qubits 8 --d 1 --qubit_constrain 5 --total_shots 10000 --num_trials 1 --with_classical_shadows