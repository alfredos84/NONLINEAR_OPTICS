# Nonlinear Optics in second-order media

## `cuOPO` package
is a C/CUDA-based toolkit for simulating optical parametric oscillators using the coupled-wave equations (CWEs) that well-describe the three wave mixing (TWM) processes in a second-order nonlinear media. CUDA programming allows you to implement parallel computing in order to speed up calculations that typically require a considerable computational demand.

The provided software implements a solver for the CWEs including dispersion terms, linear absorption and intracavity element if they are required. It also includes flags to solve nanosecond or continuous wave time regimes.

This code is useful for simulations based on three-wave mixing proccesses such as optical parametric oscillators (OPOs).
It solves the coupled-wave equations (CWEs) for signal, idler and pump using a parallel computing scheme based on CUDA programming.

For running this code is necessary to have a GPU in your computer and installed the CUDA drivers and the CUDA-TOOLKIT as well. 
To install the CUDA driver on a Linux system please visit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

### Inputs
The package provides a shell script file (`cuOPO.sh`) where a set of different relevant parameters are defined as well as some commands to create folders to save the output files and the compilation and execution lines.

 #### Compulation line
 ```
 nvcc cuOPO.cu -DCW_OPO --gpu-architecture=sm_75 -lcufftw -lcufft -o cuOPO
```
The `nvcc` compiler compiles the file `cuOPO.cu` and creates the executable object file `cuOPO`. The flags `-lcufftw` and `-lcufft` are useful for the Fourier transform performed by CUDA. The flag `-DCW_OPO` indicates the continuous-wave pumping regime (it is also available nanosecond regime by setting `-DNS-OPO`). By default, this package works at degeneracy. This means that only two equations are required. If you are working in a non-degenerate scheme, you will need three CWEs and this is passed to compilation line by adding the flag `-DTHREE_EQS`. Finally, the flag `--gpu-architecture=sm_75` is related to the GPU architecture.

#### Execution line
```
./cuOPO <LIST-OF-ARGUMENTS> | tee -a <output_file.txt>
```
This line just executes the packege and creates an output text file with the simulation outcome.

### Outputs

This package returns a set of `.dat` files with the signal, idler and pump electric fields, separated into real and imaginary parts. It also returns time and frequency vectors

### GPU architecture
Make sure you know your GPU architecture before compiling and running simulations. For example, pay special attention to the sm_75 flag defined in the provided `cuOPO.sh` file. That flag might not be the same for your GPU since it corresponds to a specific architecture. For instance, I tested this package using two different GPUs:
1. Nvidia Geforce MX250  : architecture -> Pascal -> flag: sm_60
2. Nvidia Geforce GTX1650: architecture -> Turing -> flag: sm_75

Please check the NVIDIA documentation in https://docs.nvidia.com/cuda/pascal-compatibility-guide/index.html

The provided shell script file (`cuOPO.sh`) is useful to run several simulations with different parameters by compiling just once. Those parameters could be number of round trips, crystal length, cavity length, grid size, among other relevant parameters.
