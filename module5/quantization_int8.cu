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

// Quantize Matrix multiplication, a new algo rithm where we quantise a large float value to a smaller integer value 'int_8' [8-bit signed integer]
// varies from -128 to 127
#define TILE_SIZE 16

__global__ void QuantMatmulKernel(const float* activation, const int8_t* weight_q, float scale, float* output, int N){
    // create two shared tiles i.e. WeightTile and activationTile
    __shared__ float actTile[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t weightTile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t=0; t<(N + TILE_SIZE -1)/TILE_SIZE; t++){
        int tileRow = t*TILE_SIZE + threadIdx.y;
        int tileCol = t*TILE_SIZE + threadIdx.x;
        // load the activation and the weight in to the respective tiles..
        actTile[threadIdx.y][threadIdx.x] = (row < N && tileCol < N) ? activation[row * N + tileCol]: 0.0f; // check if the row and the tileCol is <= to the N then only put the activation value (row * N + tileCol)
        weightTile[threadIdx.y][threadIdx.x] = (tileRow < N && col < N) ? weight_q[tileRow * N + col]: (int8_t)0; // chekc if the col and the tileRow is <= to the N then only put the weight_q value (tileRow * N + col)
        __syncthreads();

        for(int i=0; i<TILE_SIZE; i++){
            // have to dequantize over this place using the scale which is in signature parameter.
            float weightQuant = weightTile[i][threadIdx.x] * scale;
            sum += actTile[threadIdx.y][i] * weightQuant; 
        }
        __syncthreads();

    }
    if (row < N && col < N) output[row * N + col] = sum;
}

int main(){

    int N = 256;
    size_t actQBytes = N * N * sizeof(float);
    size_t weightQBytes = N * N * sizeof(int8_t);

    // set the activation, output and the weight(normal and quantized) matrix
    float *h_activation = new float[N*N];
    float *h_weight = new float[N*N];
    int8_t *h_weight_q = new int8_t[N*N];
    float *h_output = new float[N*N];
    // set the random number generator constant.
    srand(42);
    for (int i=0; i<N*N; i++){
        h_activation[i] = (float)(rand() % 10) / 10.0f; h_weight[i] = ((float)(rand() % 200) - 100) / 100.0f;
    }
    // quantize on the host machine, get the max number, scale it and convert it into int8 format
    float max_val = 0.0f;
    for (int i=0; i<N*N; i++){
        max_val = fmaxf(max_val, fabsf(h_weight[i]));
    }
    float scale = max_val / 127.0f; 
    // now convert it into int8 format
    for (int i=0; i<N*N; i++){
        h_weight_q[i] = (int8_t) roundf(h_weight[i] / scale);
    }

    float *d_activation, *d_output;
    int8_t *d_weight_q;
    
    CUDA_CHECK(cudaMalloc(&d_activation, actQBytes));
    CUDA_CHECK(cudaMalloc(&d_output, actQBytes));
    CUDA_CHECK(cudaMalloc(&d_weight_q, weightQBytes));

    CUDA_CHECK(cudaMemcpy(d_activation, h_activation, actQBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight_q, h_weight_q, weightQBytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

    // matmulTiled<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    QuantMatmulKernel<<<blocks, threadsPerBlock>>>(d_activation, d_weight_q, scale, d_output, N);

    CUDA_CHECK(cudaGetLastError()); // catches lauch time error
    CUDA_CHECK(cudaDeviceSynchronize()); // catches runtime error
    CUDA_CHECK(cudaMemcpy(h_output, d_output, actQBytes, cudaMemcpyDeviceToHost));

    // CPU checking, how does the cpu handles the proposed problem, but here it uses the float weight instead of the quantized weight.
    float* cpu_output = new float[N*N];
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            float sum = 0.0f;
            for (int k=0; k<N; k++){
                sum += h_activation[i*N + k] * h_weight[k*N + j];
            }
            cpu_output[i*N + j] = sum;
        }
    }

    // float max_rel_diff = 0.0f;
    // for (int i = 0; i < N*N; i++) {
    //     float rel_diff = fabsf(h_output[i] - cpu_output[i]) / (fabsf(cpu_output[i]) + 1e-6f);
    //     max_rel_diff = fmaxf(max_rel_diff, rel_diff);
    // }
    // printf("max relative diff: %f\n", max_rel_diff); 
    // printf("\nGPU c[%d][%d] = %f,\n CPU exceeded = %f\n", check_i, check_j, h_C[check_i * N + check_j], cpu_val);
    
    // lets try this one whether we get the accurate answer or not (must be ~<|>0.5)
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int num_failures = 0;
    float atol = 0.05f, rtol = 0.05f;   // tune these — quantization error scales with `scale`, so loosen if needed

    for (int i = 0; i < N*N; i++) {
        float abs_diff = fabsf(h_output[i] - cpu_output[i]);
        max_abs_diff = fmaxf(max_abs_diff, abs_diff);
        if (fabsf(cpu_output[i]) > 1.0f) {   // only compute relative diff where the denominator is meaningful
            float rel_diff = abs_diff / fabsf(cpu_output[i]);
            max_rel_diff = fmaxf(max_rel_diff, rel_diff);
        }
        if (abs_diff > atol + rtol * fabsf(cpu_output[i])) num_failures++;
    }
    printf("max abs diff: %f\n", max_abs_diff);
    printf("max rel diff (only where |cpu|>1): %f\n", max_rel_diff);
    printf("elements failing combined tolerance: %d / %d\n", num_failures, N*N);

    cudaFree(d_activation); cudaFree(d_weight_q); cudaFree(d_output); // clear the whole matrix to free memory space
    delete[] h_activation; delete[] h_weight_q; delete[] h_output; delete[] cpu_output; delete[] h_weight;

}

// NOTE FOR THE DEV:

//  - here the error occurs even though it passed the "nvcc .\quantization_int8.cu -o quantizeInt8 -arch=sm_86" as nothing wrong. because of the un hindered memory bound
//  in the matrix (out of bound error can leads to the cause of silent error)
//  - properly check the collon at the end ';'  
//  - have the skill to learn "how to resolve errors without getting frustrated". (i got a lots of error too, but solved one by one)
//