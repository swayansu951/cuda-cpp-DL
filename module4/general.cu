// In this we will see how we always check the CUDA error which catches any kind of cuda error before compiling to the binary exe file
// the main thing is the CUDA_CHECK

#include <cstdio>
#include <cstdlib>
#include <cmath>

const int N = 16;

dim3 threadsPerBlock(16,16); // 256 threads per block (16 * 16)
dim3 blocks((N+15) / 16, (N+15)/16); // blocks to cover enough N * N matrix

matmulKernel<<<blocks, threadsPerBlock>>>(A, B, C, N);

// the same as row * N + col: 
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Alway check the CUDA error:
#define CUDA_CHECK(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess){\
        fprintf(stderr, "Cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);\
    }\
}