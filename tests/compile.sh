#!/bin/bash
nvcc tensornet_example.cu -I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include -L${CUQUANTUM_ROOT}/lib -L${CUTENSOR_ROOT}/lib -lcutensornet -lcutensor -o tensornet_example