// the work flow in this is like, the d_x (dimention is of [4, 384]) goes to the ----> layerNorm (ln1), 
// then the output from it...
// d_ln (dimention is of [3, 384]) goes to the --------> matmulKernel (with attn_qkv), whose output is
// generated -----> 
// d_qkv..
// (dimention is of [4, 1152])

// here we will 1st initiate the layerNorm code (which we did before) and the matmul kernel too..

// layerNorm implementation: 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../6a/layerNormKernel.cu"
#include "../6b/FeedForward.cu"
#include "../6b/fnn_full.cu"

// okay we have both of the functions, we will call them and pass the paramter direcltly to them. 

// change from here...
int main(){
    int hiddenDim = 384;
    float eps = 1e-5f;
    int seqlen = 4;
    int dim = 1152;
    // for this the beta remain the zero as no bias
    size_t extraParam = hiddenDim * sizeof(float);
    size_t inOpParam = 4 * hiddenDim * sizeof(float);
    size_t outParam = 4 * 1152 * sizeof(float);

    float *d_ln, *d_x, *d_zero, *d_qkv;
    CUDA_CHECK(cudaMalloc(&d_ln, inOpParam));
    CUDA_CHECK(cudaMalloc(&d_qkv, outParam));
    CUDA_CHECK(cudaMalloc(&d_zero, 384*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, inOpParam));
    CUDA_CHECK(cudaMemset(d_zero, 0, 384 * sizeof(float)));

    // CUDA_CHECK(cudaMemcpy(d_input, h_input, inOpParam, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_beta, beta, extraParam, cudaMemcpyHostToDevice));

    int threadPerBlock = 256;
    size_t sharedMemBytes = threadPerBlock * sizeof(float);
    
    //embed kernel to get the d_x from it
    embedKernel<<<4, 384>>>(d_x, hiddenDim);
    CUDA_CHECK(cudaGetLastError()); // catches lauch time error
    CUDA_CHECK(cudaDeviceSynchronize());

    LayerNormKernel<<<seqlen, threadPerBlock, sharedMemBytes>>>(d_x, d_layer[0].ln1_gamma, d_zero, d_ln, seqlen, hiddenDim, eps);
    CUDA_CHECK(cudaGetLastError()); // catches lauch time error
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((1152 + TILE_SIZE - 1)/TILE_SIZE, (4 + TILE_SIZE - 1)/TILE_SIZE);
    matmulKernel <<<grid, threads>>>(d_ln, d_layer[0].attn_qkv, d_qkv, seqlen, dim, hiddenDim);
    CUDA_CHECK(cudaGetLastError()); // catches lauch time error
    CUDA_CHECK(cudaDeviceSynchronize()); // catches runtime error
    // CUDA_CHECK(cudaMemcpy(h_output, d_ln, inOpParam, cudaMemcpyDeviceToHost)); we don't need this now 

    cudaFree(d_qkv); cudaFree(d_x); cudaFree(d_zero); cudaFree(d_ln);// clear the whole matrix to free memory space
    delete[] beta; delete[] h_input; delete[] h_output;
}



