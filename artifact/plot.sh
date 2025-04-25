#!/bin/bash

# Run the first Python script
python plot.py

# Change directory to sparse_contract_examples
cd ../sparse_contract_examples && ls
python plot.py
# Copy the specified files to ../artifact/figs
cp -r figure/ ../artifact/figs/sec_v_figs