# Parallel Optimization of NTT for Kyber Key Encapsulation Mechanism using GPU Accelerated Plantard Arithmetic

## Overview
This repository contains the source code for the GPU-optimized implementation of the Number Theoretic Transform (NTT) using Plantard arithmetic for the Kyber Key Encapsulation Mechanism (KEM). The implementation is designed to improve performance by avoiding warp divergence and reducing expensive division operations.

## Features
- Optimized GPU-based implementation of Plantard arithmetic for Kyber KEM.
- Techniques to minimize warp divergence and improve computational efficiency.

## Requirements
- **CUDA Toolkit** (Ensure that the appropriate version is installed)
- **NVIDIA GPU** (RTX 4080 or equivalent recommended for best performance)
- **C++ Compiler** (G++ recommended)
- **Make**

## Repository Structure
```
├── src/               # Source code directory
│   ├── main.c        # Main execution file
│   ├── sha2.c        # SHA-2 implementation
│   ├── incdpa.c      # Indeterminate Domain Polynomial Arithmetic
│   ├── poly_func.c   # Polynomial functions
│   ├── cuda_kernel.cu # CUDA kernel implementation
│   ├── reduce.c      # Reduction functions
│   ├── aes_gpu.c     # AES implementation on GPU
├── include/          # Header files
│   ├── sha2.h
│   ├── incdpa.h
│   ├── poly_func.h
│   ├── cuda_kernel.cuh
│   ├── reduce.h
│   ├── aes_gpu.h
├── bin/               # Compiled object files
├── Makefile           # Compilation instructions
└── README.md          # Documentation (this file)
```

## Compilation
To compile the project, ensure that CUDA is properly installed and configured. Run the following command in the root directory:
```bash
make
```
This will generate the `run_test` executable.

## Execution
After compilation, execute the binary using:
```bash
./run_test
```

## Makefile Explanation
The provided `Makefile` automates the compilation process. It includes:
- **CUDA Paths:** Ensures that the compiler can locate CUDA headers and libraries.
- **Compiler Options:** Uses `g++` for C++ files and `nvcc` for CUDA files.
- **Optimization Flags:** Includes `-arch=sm_86` for compatibility with modern GPUs.
- **Dependency Management:** Handles compilation of `.c` and `.cu` files separately.

To clean the compiled files:
```bash
make clean
```

## Acknowledgments
Special thanks to **Wai-Kong Lee** ([GitHub Profile](https://github.com/benlwk)) for his valuable help with the code implementation.
