#!/bin/bash

mkdir ../dataset
python test_postprocessing.py
# Navigate to the sparse_contraction_example directory
cd ../sparse_contract_examples

# Run the Python script
python run.py