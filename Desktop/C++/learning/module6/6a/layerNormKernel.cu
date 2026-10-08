#include <cstdio>
#include <cstdlib>
#include <cmath>

// Alway check the CUDA error:
#define CUDA_CHECK(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess){\
        fprintf(stderr, "Cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);\
    }\
}

__global__ void LayerNormKernel(const float* input, const float* gamma, const float* beta, float* output, int seqLen, int hiddenDim, float eps){
    int row = blockIdx.x; // one block per token 
    int tid = threadIdx.x;
    // a shared matrix as sdata[]
    extern __shared__ float sdata[];
    // and a row pointer to read the value only
    const float* rowPtr = input + row * hiddenDim;

    // get the local sum and store it in the shared data matrix.
    float local_sum = 0.0f;
    for(int i=tid; i < hiddenDim; i+= blockDim.x){
        local_sum += rowPtr[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();
    // get the mean value of each block.
    // tree reduction
    for (int i=blockDim.x/2; i > 0; i>>= 1){
        if (tid < i) sdata[tid] += sdata[tid+i]; // until the thread remains under the bound we have to add those values with the next one
        __syncthreads();
    }
    float mean = sdata[0] / hiddenDim;
    __syncthreads();
    
    float local_sq_sum = 0.0f;
    for(int i=tid; i < hiddenDim; i+= blockDim.x){
        float diff = rowPtr[i] - mean;
        local_sq_sum += diff * diff;
    }
    sdata[tid] = local_sq_sum;
    __syncthreads();

    // now get the variance
    for (int i=blockDim.x/2; i>0; i>>=1){
        if (tid < i) sdata[tid] += sdata[tid + i];
        __syncthreads();
    }
    float variance = sdata[0]/hiddenDim;
    __syncthreads();

    // now each thread writes it's own output
    for (int i=tid; i<hiddenDim; i+=blockDim.x){
        output[row * hiddenDim + i] = (rowPtr[i] - mean) / sqrtf(variance + eps) * gamma[i] + beta[i];
    }
}

int main(){
    int hiddenDim = 256;
    float eps = 1e-5f;
    int seqlen = 16;
    float *gamma = new float[hiddenDim];
    float *beta = new float[hiddenDim];
    float *h_input = new float[seqlen * hiddenDim];
    float *h_output = new float[seqlen * hiddenDim];
    size_t extraParam = hiddenDim * sizeof(float);
    size_t inOpParam = seqlen * hiddenDim * sizeof(float);

    srand(42);
    // we initialize some random values to the gamma and the beta
    for (int i=0; i<hiddenDim; i++){
        gamma[i] = (float)(rand() % 10) / 10.0f;
        beta[i] = (float)(rand() % 10) / 10.0f; 
    }
    // same we have to put random value to the h_input
    for (int i=0; i<seqlen*hiddenDim; i++){
        h_input[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    }
    float *d_output,  *d_input, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_output, inOpParam));
    CUDA_CHECK(cudaMalloc(&d_input, inOpParam));
    CUDA_CHECK(cudaMalloc(&d_gamma, extraParam));
    CUDA_CHECK(cudaMalloc(&d_beta, extraParam));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, inOpParam, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma, extraParam, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta, extraParam, cudaMemcpyHostToDevice));

    int threadPerBlock = 256;
    size_t sharedMemBytes = threadPerBlock * sizeof(float);

    LayerNormKernel<<<seqlen, threadPerBlock, sharedMemBytes>>>(d_input, d_gamma, d_beta, d_output, seqlen, hiddenDim, eps);

    CUDA_CHECK(cudaGetLastError()); // catches lauch time error
    CUDA_CHECK(cudaDeviceSynchronize()); // catches runtime error
    CUDA_CHECK(cudaMemcpy(h_output, d_output, inOpParam, cudaMemcpyDeviceToHost));

    // now we take the cpu reference, how does the output differs and how much..
    float* cpu_output = new float[seqlen * hiddenDim];
    for(int i=0; i<seqlen; i++){
        const float* r = h_input + i * hiddenDim;
        float mean = 0.0f;
        for (int l=0; l<hiddenDim; l++) mean += r[l];
        mean /= hiddenDim;

        float var = 0.0f;
        for (int j=0; j<hiddenDim; j++) var += (r[j] - mean) * (r[j] - mean);
        var /= hiddenDim;

        for (int k=0; k<hiddenDim; k++){
            cpu_output[i*hiddenDim + k] = (r[k] - mean) / sqrtf(var + eps) * gamma[k] + beta[k];
        }

    }
    // now get the max difference between the gpu result and the cpu result
    float max_diff = 0.0f;
    for (int i=0; i<seqlen*hiddenDim; i++){
        max_diff = fmaxf(max_diff, fabsf(h_output[i] - cpu_output[i]));
    }
    printf("The max CPU and GPU output is : %f\n", max_diff);

    cudaFree(d_beta); cudaFree(d_gamma); cudaFree(d_output); // clear the whole matrix to free memory space
    delete[] beta; delete[] gamma; delete[] h_input; delete[] h_output; delete[] cpu_output;
}



