
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

#define CUDA_CHECK(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess){\
        fprintf(stderr, "Cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);\
    }\
}

float* loadWeight(const char* filename, size_t numFloat){
    float* buf = new float[numFloat];
    FILE* f = fopen(filename, "rb");

    if (f == nullptr) {
        fprintf(stderr, "Failed to open: %s\n", filename);
        exit(1);
    }
    // size mismatch check..
    // check if the last element of the file is not zero, if zero then nothing, its empty..
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t expectedSize = numFloat * sizeof(float);

    if ((size_t)size != expectedSize){
        fprintf(stderr, "Size mismatch: %s has %d bytes, expected %zu\n", filename, size, expectedSize);
        exit(1);
    }
    
    // go to the file 'f' and read a total numFloat item where each element 
    // is of float and copy those bytes and store in a memroy buffer 'buf' 
    // if the content inside of the file is not as we expected, we have to check that too
    size_t readContent = fread(buf, sizeof(float), numFloat, f);
    fclose(f);

    if (readContent != numFloat){
        fprintf(stderr, "Short content read: %s got %zu, but we expected %zu\n", filename, readContent, numFloat);
        exit(1);
    }

    return buf;
}

// then call the above function for all the file inside the folder 'weight'..
// we know that both wte and lm_head share memory.
struct layerWeights{
    float* ln1_gamma;
    float* ln2_gamma;
    float* attn_proj;
    float* attn_qkv;
    float* mlp_fc;
    float* mlp_proj;
    
};

// EmbedKernel
// a global function that takes the read only matrix tokenIds, wte and wpe as input, hiddenDim cosntant int value
// it maps every sequence of token into continuous vector, for each token position it calculates the final embedding vector by adding the token embedding to its 
// positional embedding   
__global__ void embedKernel(const int* tokenIds, const float* wte, const float* wpe, float* output, int hiddenDim){
    // hiddenDim is of 386
    int token = blockIdx.x;
    int i = threadIdx.x;
    
    int tId = tokenIds[token];
    output[token * hiddenDim + i] = wte[tId * hiddenDim + i] + wpe[token * hiddenDim + i];
}


// matmulKernel
__global__ void matmulKernel(const float* A, float* B, float* C, int M, int N, int K){
    // here we will set the shared matrix of tile size of data type float..
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Compute the thread's row and column..
    // the same as row * N + col: 
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // now instead of doing blockDim we will use the tile_size that we defined above
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // if ((row >= TILE_SIZE) && (col >= TILE_SIZE)){
    //     printf("Exced the limit! row : %d  |  col : %d", row, col);
    //     return;
    // }

    float sum = 0.0f;
    // so instead of using the limit as N we will now  we the (N + TILE_SIZE - 1)/ TILE_SIZE
    for (int t=0; t<(K + TILE_SIZE - 1)/TILE_SIZE; t++){
        // Now each thread load one element into shared memory
        int tileCol = t * TILE_SIZE + threadIdx.x;
        int tileRow = t * TILE_SIZE + threadIdx.y;

        // Now we have to check whether the row and column is not exceed the global matrix. This prevent when the matrix size is 
        // not evenly divided by the block size.
        // if not exceed (True) then we will load the value from the global memory arrya 'A' using 1D row-major indexing (row * N + tiledCol)
        // if not then padded by 0.0f.
        // for A the matrix is A[m,k]
        // and for B the matrix is B[k,n]
        // and for the output matrix is, C[m,n]
        tileA[threadIdx.y][threadIdx.x] = (row < M && tileCol < K) ? A[row * K + tileCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tileRow < K && col < N) ? B[tileRow * N + col] : 0.0f;
        
        __syncthreads(); // wait until all the threads in the block finishes loading 

        for (int i = 0; i < TILE_SIZE; i++){
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        
        __syncthreads(); // wiat until all finishes done reading before the next tile overwrites
    }
    // at the end check if the row and col not exceed the N
    if (row < M && col < N){
        C[row*N + col] = sum;
    }

}

// layerNormKernel
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

float* toDevice(const float* host, size_t numFloat){
    // instead of manually repeat the task we did before i.e. the coping cpu memeroy data to the GPU, we will do this in this once.
    size_t bytes = numFloat * sizeof(float);
    float* device = nullptr;
    // now we use this function to create location in the GPU, and then share the value to the GPU memory (destination, source, bytes, direction)
    CUDA_CHECK(cudaMalloc(&device, bytes));
    CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));

    return device;
}

__global__ void addKernel(float* x, const float* y, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

int main(){
    layerWeights layer[6];
    layerWeights d_layer[6];

    for (int i=0; i<6; i++){
        char path[256];

        sprintf(path, "weights/h%d_ln1_gamma.bin", i);
        layer[i].ln1_gamma = loadWeight(path, 384);

        sprintf(path, "weights/h%d_ln2_gamma.bin", i);
        layer[i].ln2_gamma = loadWeight(path, 384);
        
        sprintf(path, "weights/h%d_attn_qkv.bin", i);
        layer[i].attn_qkv = loadWeight(path, 384 * 1152);

        sprintf(path, "weights/h%d_attn_proj.bin", i);
        layer[i].attn_proj = loadWeight(path, 384 * 384);

        sprintf(path, "weights/h%d_mlp_fc.bin", i);
        layer[i].mlp_fc = loadWeight(path, 384 * 1536);

        sprintf(path, "weights/h%d_mlp_proj.bin", i);
        layer[i].mlp_proj = loadWeight(path, 1536 * 384);

    }
    // one fucntion instead of all manual input and managing hectic function parameters too..
    float* h_wte = loadWeight("weights/wte.bin", 65 * 384);
    float* h_lm_head = loadWeight("weights/lm_head.bin", 65 * 384);
    float* h_lnf_gamma = loadWeight("weights/lnf_gamma.bin", 384);
    float* h_wpe = loadWeight("weights/wpe.bin", 256 * 384);

    for (int i=0; i<6; i++){
        d_layer[i].ln1_gamma = toDevice(layer[i].ln1_gamma, 384);
        d_layer[i].ln2_gamma = toDevice(layer[i].ln2_gamma, 384);
        d_layer[i].attn_qkv = toDevice(layer[i].attn_qkv, 384 * 1152);
        d_layer[i].attn_proj = toDevice(layer[i].attn_proj, 384 * 384);
        d_layer[i].mlp_fc = toDevice(layer[i].mlp_fc, 384 * 1536);
        d_layer[i].mlp_proj = toDevice(layer[i].mlp_proj, 1536 * 384);

    }
    float* d_wte = toDevice(h_wte, 65 * 384);
    float* d_lm_head = toDevice(h_lm_head, 65 * 384);
    float* d_lnf_gamma = toDevice(h_lnf_gamma, 384);
    float* d_wpe = toDevice(h_wpe, 256 * 384);
    float* h_qkv = new float[4 * 1152];

    int h_token[5] = {5,10,20,30,1};
    float h_x[4*384];

    float *d_x, *d_ln, *d_zero, *d_qkv;
    int *d_token;

    CUDA_CHECK(cudaMalloc(&d_x, 4*384*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_token, 4*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_token, h_token, 4*sizeof(int), cudaMemcpyHostToDevice));

    int hiddenDim = 384;
    float eps = 1e-5f;
    int M = 4;
    int seqlen = 6;
    int dim = 1152;
    // for this the beta remain the zero as no bias
    size_t extraParam = hiddenDim * sizeof(float);
    size_t inOpParam = 4 * hiddenDim * sizeof(float);
    size_t outParam = 4 * 1152 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_ln, inOpParam));
    CUDA_CHECK(cudaMalloc(&d_qkv, outParam));
    CUDA_CHECK(cudaMalloc(&d_zero, 384*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_zero, 0, 384 * sizeof(float)));

    embedKernel<<<4, 384>>>(d_token, d_wte, d_wpe, d_x, 384);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // collects the result from the gpu back to the cpu
    CUDA_CHECK(cudaMemcpy(h_x, d_x, 4*384*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Result from EmbedKernel\n1. %f, 2. %f, 3. %f \n", h_x[0], h_x[1], h_x[2]);

    int threadPerBlock = 256;
    size_t sharedMemBytes = threadPerBlock * sizeof(float);
    LayerNormKernel<<<M, threadPerBlock, sharedMemBytes>>>(d_x, d_layer[0].ln1_gamma, d_zero, d_ln, M, hiddenDim, eps);
    CUDA_CHECK(cudaGetLastError()); // catches lauch time error
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((1152 + TILE_SIZE - 1)/TILE_SIZE, (4 + TILE_SIZE - 1)/TILE_SIZE);
    matmulKernel <<<grid, threads>>>(d_ln, d_layer[0].attn_qkv, d_qkv, M, dim, hiddenDim);
    CUDA_CHECK(cudaMemcpy(h_qkv, d_qkv, 4*1152*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Result from matmulkerel\n1st three vlaues(host side): h_qkv[0]: %f, h_qkv[1]: %f, h_qkv[2]: %f \n", h_qkv[0], h_qkv[1], h_qkv[2]);
    CUDA_CHECK(cudaGetLastError()); // catches lauch time error
    CUDA_CHECK(cudaDeviceSynchronize()); // catches runtime error

    cudaFree(d_qkv); cudaFree(d_x); cudaFree(d_zero); cudaFree(d_ln); cudaFree(d_wte); cudaFree(d_wpe); cudaFree(d_lm_head); cudaFree(d_lnf_gamma);
    delete[] h_wte; delete[] h_wpe; delete[] h_lm_head; delete[] h_lnf_gamma; delete[] h_qkv;
        
    for(int i=0; i<6; i++){
        delete[] layer[i].attn_proj;
        delete[] layer[i].attn_qkv;
        delete[] layer[i].ln1_gamma;
        delete[] layer[i].ln2_gamma;
        delete[] layer[i].mlp_fc;
        delete[] layer[i].mlp_proj;
    }
    for(int i=0; i<6; i++){
        cudaFree(d_layer[i].attn_proj);
        cudaFree(d_layer[i].attn_qkv);
        cudaFree(d_layer[i].ln1_gamma);
        cudaFree(d_layer[i].ln2_gamma);
        cudaFree(d_layer[i].mlp_fc);
        cudaFree(d_layer[i].mlp_proj);
    }
}
