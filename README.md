# CUDA-CPP

- A structural repo documentary of learning CUDA programming , GPU computing, and DL implementation from sratch.
- In this repo we will learn how the deeplearning libraries are being used, instead of using them directly we gonna write them from scratch and compare the performance with the python library (pytorch for this case).
- Here we have covered from the basic CPP required and a slow dive into the Cuda-CPP indepth.

![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![C++](https://img.shields.io/badge/C++-17-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

## About

This repository contains my notes, implementations,
and experiments while learning

- CUDA Programming
- GPU Architecture
- Parallel Computing
- CUDA Kernels
- CUDA Memory
- Performance Optimization
- Deep Learning from Scratch

Instead of only using PyTorch or TensorFlow,
I want to understand how they work internally.

## Repo Structure
```md
cuda-cpp-DL/
    ├── 01_Basics/
    ├── 02_Memory/
    ├── 03_Parallelism/
    ├── 04_Optimization/
    ├── 05_DeepLearning/
    ├── notes/
    ├── images/
    └── README.md
```

## Instructions
```md
git clone https://github.com/swayansu951/cuda-cpp-DL.git

cd cuda-cpp-DL

nvcc vector_add.cu -o vector_add

./vector_add
```

## Resources

Some resources you can follow:

- CUDA Programming Guide
- CUDA Best Practices
- PTX Documentation
- NVIDIA Developer Blogs

## Roadmap 
- For your journey, you can follow this path too:
    
    - [ ] CUDA Installation
    - [ ] First CUDA Program
    - [ ] Matrix Multiplication
    - [ ] Thread Hierarchy
    - [ ] Shared Memory
    - [ ] Activation functions 
    - [ ] Streams
    - [ ] Reduction
    - [ ] CNN Kernels
    - [ ] Transformers

## Topics covered
- CUDA Runtime API
- CUDA Driver API
- Thread Blocks
- Grids
- Warps
- Shared Memory
- Global Memory
- Constant Memory
- Streams
- Events
- Occupancy
- Tensor Cores
- GEMM
- CNN
- Attention

## License
MIT License