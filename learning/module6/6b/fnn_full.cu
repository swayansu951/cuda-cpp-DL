// reads the file (to get the content)
//signature (to get the weights of the .pt file), filename and the size of float value (n)
// create a n size of float array 
// read the file and get the content in a variable
// open the binary file in read only mode
// WTE:: word token embedding
// WPE:: word position embedding

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>


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

#define CUDA_CHECK(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess){\
        fprintf(stderr, "Cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);\
    }\
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
    // layerWeights d_layer[6];

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


    // for (int i=0; i<6; i++){
    //     d_layer[i].ln1_gamma = toDevice(layer[i].ln1_gamma, 384);
    //     d_layer[i].ln2_gamma = toDevice(layer[i].ln2_gamma, 384);
    //     d_layer[i].attn_qkv = toDevice(layer[i].attn_qkv, 384 * 1152);
    //     d_layer[i].attn_proj = toDevice(layer[i].attn_proj, 384 * 384);
    //     d_layer[i].mlp_fc = toDevice(layer[i].mlp_fc, 384 * 1536);
    //     d_layer[i].mlp_proj = toDevice(layer[i].mlp_proj, 1536 * 384);

    // }
    float* d_wte = toDevice(h_wte, 65 * 384);
    float* d_lm_head = toDevice(h_lm_head, 65 * 384);
    float* d_lnf_gamma = toDevice(h_lnf_gamma, 384);
    float* d_wpe = toDevice(h_wpe, 256 * 384);

    int h_token[5] = {5,10,20,30,1};
    float h_x[4*384];

    float *d_x;
    int *d_token;

    CUDA_CHECK(cudaMalloc(&d_x, 4*384*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_token, 4*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_token, h_token, 4*sizeof(int), cudaMemcpyHostToDevice));

    embedKernel<<<4, 384>>>(d_token, d_wte, d_wpe, d_x, 384);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // collects the result from the gpu back to the cpu
    CUDA_CHECK(cudaMemcpy(h_x, d_x, 4*384*sizeof(float), cudaMemcpyDeviceToHost));
    printf("1. %f, 2. %f, 3. %f", h_x[0], h_x[1], h_x[2]);

    cudaFree(d_wte); cudaFree(d_wpe); cudaFree(d_lm_head); cudaFree(d_lnf_gamma);
    // for(int i=0; i<6; i++){
    //     cudaFree(d_layer[i].attn_proj);
    //     cudaFree(d_layer[i].attn_qkv);
    //     cudaFree(d_layer[i].ln1_gamma);
    //     cudaFree(d_layer[i].ln2_gamma);
    //     cudaFree(d_layer[i].mlp_fc);
    //     cudaFree(d_layer[i].mlp_proj);
    // }
    delete[] h_wte; delete[] h_wpe; delete[] h_lm_head; delete[] h_lnf_gamma;
    for(int i=0; i<6; i++){
        delete[] layer[i].attn_proj;
        delete[] layer[i].attn_qkv;
        delete[] layer[i].ln1_gamma;
        delete[] layer[i].ln2_gamma;
        delete[] layer[i].mlp_fc;
        delete[] layer[i].mlp_proj;
    }
}
