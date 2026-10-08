#include <cstdio>

// A simple cuda-CPP code before dive into deeper, we will always use this '__global__' specifier to let the CPU to use and instruct the GPU.
__global__ void addKernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1024;
    size_t bytes = n * sizeof(float);

    float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
    for (int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = i * 2; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    // this below code tells that the total blocks to use, the threads Per Block to use from the GPU (in total we use 3 such instructions)
    addKernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("c[0]=%f c[1023]=%f (expect 0 and 3069)\n", h_c[0], h_c[1023]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
}

// test it now ! :)