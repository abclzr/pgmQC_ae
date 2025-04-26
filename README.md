# pgmQC

### installation
```
pip install -r requirements.txt
pip install --editable .
```

### Compilation (Optional)
The compiled executable files are provided. You can skip the compilation if you didn't modified any cuda or c++ source code.
```
cd cuquantum_examples
nvcc build_task.cu -I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include -L${CUQUANTUM_ROOT}/lib -L${CUTENSOR_ROOT}/lib -lcutensornet -lcutensor -o build_task -O3
nvcc run_cuquantum_contract.cu -I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include -L${CUQUANTUM_ROOT}/lib -L${CUTENSOR_ROOT}/lib -lcutensornet -lcutensor -o run_cuquantum_contract -O3
cd ../sparse_contract_examples
make
cd ..
```

### Workflow
first change directory to the ```artifact/```.
```
cd artifact
```
#### Section IV
run pairwise connected HWEA (~1 hour)
```
bash sec_iv_HWEA.sh
```
run 3local HWEA (~1 hour)
```
bash sec_iv_3local.sh
```
run QNN example (~12 hours)
```
bash sec_iv_qnn.sh
```
compare with classical shadows (~2 hours)
```
bash sec_iv_classical_shadows.sh
```

#### Section V
prepare tensor contraction workloads and run (~0.5 hours)
```
bash sec_v_run.sh
```
plot all figures and save in ```artifact/figs/```
```
bash plot.sh
```
<!-- ### Run qft_6 for example
```
cd ../cuquantum_examples
./build_task ../dataset/qft_6 ../dataset/qft_6_subgraph1 ../dataset/qft_6_subgraph2
./run_cuquantum_contract ../dataset/qft_6
``` -->
