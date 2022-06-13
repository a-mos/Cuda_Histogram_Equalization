# Cuda Histogram Equalization
Histogram equalization on CUDA from scratch, benchmarking CPU with OpenMP performance vs CUDA
<img src="WasteWhite_LDR_0001.png" width="400"><img src="out_gpu.png" width="400">

## Requirements
Tested with CUDA 11.6, NVCC 11.6, Ubuntu 20.04

## Compile
    cd project_folder
    nvcc -Xcompiler -fopenmp -O2 kernel.cu main.cpp -o filter

## Run
    ./filter {path_to_image}

## Examples

`./filter Bathroom_LDR_0001.png`  
Img shape: 4096x4096  
Filter CPU time (num threads-1): 214.736 ms  
Filter CPU time (num threads-4): 163.62 ms  
Filter CPU time (num threads-8): 134.421 ms  
Filter GPU time (without copy): 1.05478 ms  
Filter GPU time (with copy): 21.5437 ms  
GPU Copy time: 20.4889 ms  
  
`./filter WasteWhite_LDR_0001.png`   
Img shape: 1459x2048  
Filter CPU time (num threads-1): 36.7082 ms   
Filter CPU time (num threads-4): 25.2879 ms   
Filter CPU time (num threads-8): 18.1174 ms   
Filter GPU time (without copy): 0.244816 ms  
Filter GPU time (with copy): 4.29588 ms   
GPU Copy time: 4.05107 ms  


## Optimizations
* Using shared memory as histogram storage [nvidia blog](https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
* Parallel reduction using SCAN alghoritm [for more details](http://lumetta.web.engr.illinois.edu/408-S20/slide-copies/ece408-lecture16-S20.pdf)
* CPU version parallelized using OpenMP 
