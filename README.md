# pgmQC

### installation
```
pip install -r requirements.txt
pip install --editable .
```


### Workflow
first change directory to the ```artifact/```.
```
cd artifact
```
For artifact evaluators, to make your life easier, you can run ```bash sec_iv_HWEA.sh; bash sec_iv_3local.sh; bash sec_iv_qnn.sh; bash sec_iv_classical_shadows.sh; python plot.py``` instead of seperately.
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
Plot all figures of section iv.
```
python plot.py
```
#### Section V
##### Installation of cuQuantum
The installation of cuda is a little bit tricky and requires root permission. I recommend skipping the evaluation of section v if the evaluators don't have cuda compilation and runtime libraries off the shelf and not familiar with cuda installation and compilation.

For compilation and run on GPU, make sure you have an Nvidia GPU and cuda installed, so that you can use ```nvcc``` to compile and have necessary cuda runtime libraries like cublas to make it runable on Nvidia GPUs.

Then, you need to install cuQuantum.
```
pip install cuquantum==24.3.0.post1
```
Use ```pip show cuquantum``` to find the location, name it ```path_to_packages```, and set the environment variable ```CUQUANTUM_ROOT``` and ```CUTENSOR_ROOT``` correctly.
```
export CUQUANTUM_ROOT=path_to_packages/cuquantum
export CUTENSOR_ROOT=path_to_packages/cutensor
```
Also append the paths to libraries like cutensornet, cutensor to ```LD_LIBRARY_PATH```, so that the ```-lcutensornet -lcutensor``` in compilation can be found.
```
export LD_LIBRARY_PATH=${CUQUANTUM_ROOT}/lib:${CUTENSOR_ROOT}/lib:${LD_LIBRARY_PATH}
```
Finally, compile.
```
cd cuquantum_examples
nvcc build_task.cu -I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include -L${CUQUANTUM_ROOT}/lib -L${CUTENSOR_ROOT}/lib -lcutensornet -lcutensor -o build_task -O3
nvcc run_cuquantum_contract.cu -I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include -L${CUQUANTUM_ROOT}/lib -L${CUTENSOR_ROOT}/lib -lcutensornet -lcutensor -o run_cuquantum_contract -O3
cd ../sparse_contract_examples
make
cd ..
```
#### Workflow
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
