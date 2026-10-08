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
#define TILE_SIZE  16
// gelu kernel has parameters, a matrix x and a hiddenDim n
__global__ void GeluKernel(float* x, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        float var = x[i];
        // get the value num * (var + num2 * var^3)
        float inner = 0.7978845608f * (var + 0.044715f * var*var*var);
        x[i] = 0.5f * var * (1.0f + tanh(inner)); // add this to the matrix back x[i]
    }   
}

// signature parameters, A and B are matrix, C is the output and M, N, k are the number to make the matrix
/*@devdoc
description: takes 2 different matrixes, with hiddenDim and a seqlen and compute it to 1st matmul kernel and then to GELU kernel
params: None
return: float  
@enddevdoc*/
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

int main(){
    int seqlen = 16;
    int hiddenDim = 256;
    int m1 = seqlen;
    int k1 = hiddenDim;
    int n1= 4*hiddenDim;

    int n2 = hiddenDim;
    int k2 = 4*hiddenDim;
    int m2 = seqlen;

    size_t extraParam = seqlen * hiddenDim * sizeof(float);
    size_t inOpParam = hiddenDim * (4 * hiddenDim) * sizeof(float);
    size_t hiddenParam = seqlen * (4*hiddenDim) * sizeof(float);

    // the size of the w1 and the w2 must be the same as (hiddenDim, 4*hiddenDim) (opposite for the w2)
    // all other matrix should must be of size (seqlen, hiddenDim)
    float *h_w1 = new float[hiddenDim * (4*hiddenDim)], *h_x = new float[seqlen * hiddenDim];
    float *h_w2 = new float[(4*hiddenDim) * hiddenDim], *h_hidden = new float[seqlen * (4*hiddenDim)], *h_output = new float[seqlen * hiddenDim];
    // create random value for the matrixes
    srand(42);
    for (int i=0; i<seqlen*hiddenDim; i++){
        h_x[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }
    for (int i=0; i<hiddenDim*(4*hiddenDim); i++){
        h_w1[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }
    for (int i=0; i<hiddenDim*(4*hiddenDim); i++){
        h_w2[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }

    float *d_x, *d_w1, *d_w2, *d_hidden, *d_output;
    // reserve that many bytes on GPU, and writes the resulting into the respective address.
    // double check if for all the inputs we have alocated gpu memory or not.
    CUDA_CHECK(cudaMalloc(&d_w1, inOpParam));
    CUDA_CHECK(cudaMalloc(&d_w2, inOpParam));
    CUDA_CHECK(cudaMalloc(&d_x, extraParam));
    CUDA_CHECK(cudaMalloc(&d_hidden, hiddenParam));
    CUDA_CHECK(cudaMalloc(&d_output, extraParam));

    // copy the device pointer(GPU) from the host pointer(CPU), copies physically (pcie bus) from CPU array to GPU buffer.
    // double check if for all inputs we have created cudaMemcpy
    CUDA_CHECK(cudaMemcpy(d_x, h_x, extraParam, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1, inOpParam, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2, inOpParam, cudaMemcpyHostToDevice));

    // it must be a 2D thread of size (16,16)
    // and a 2D grid block of size (M,N) like, (M*Tile_size -1)/ tile_size
    dim3 block1((n1 + TILE_SIZE - 1) / TILE_SIZE, (m1 + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threadPerBlock1(TILE_SIZE, TILE_SIZE);
    // only use the sharedMemByte where we declared a shared matrix using 'extern' keyword
    // but here we have decalred a tight shared matrix size of [tile_size][tile_size]
    // size_t sharedMemBytes = threadPerBlock * sizeof(float);
    // 1st matmul tiled kernel gets the input and gives the output
    matmulKernel<<<block1, threadPerBlock1>>>(d_x, d_w1, d_hidden, m1, n1, k1);
    CUDA_CHECK(cudaGetLastError());

    int threadPerBlock2 = 256;
    int totalElement = seqlen * (4*hiddenDim);
    int block2 = (totalElement + (threadPerBlock2-1)) / threadPerBlock2;
    // that output will be the input for the gelukernel
    GeluKernel<<<block2, threadPerBlock2>>>(d_hidden, totalElement);
    CUDA_CHECK(cudaGetLastError());
    
    dim3 block3((n2 + TILE_SIZE - 1) / TILE_SIZE, (m2 + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threadPerBlock3(TILE_SIZE, TILE_SIZE);
    // after this the final output will again goes to the matmulkernel (tiled)
    matmulKernel<<<block3, threadPerBlock3>>>(d_hidden, d_w2, d_output, m2, n2, k2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize())
    CUDA_CHECK(cudaMemcpy(h_output, d_output, extraParam, cudaMemcpyDeviceToHost));
    
    // write the max_difference, run in the CPU and subtract the result with the GPU perspective and show the result...
    // use the 'h_' suffix things only..
    float* cpu_output = new float[seqlen * hiddenDim];
    float* cpu_hidden = new float[seqlen * (4*hiddenDim)];
    float max_diff = 0.0f;
    // matmul part for CPU
    for (int row = 0; row<m1; row++){
        for (int col = 0; col<n1; col++){
            float sum = 0.0f;
            for(int K = 0; K < k1; K++){
                sum += h_x[row*k1 + K] * h_w1[K*n1 + col];
            }
            cpu_hidden[row * n1 + col] = sum;
        }
    }
    // GELU part for the CPU
    for (int i=0; i<totalElement; i++){
        float var = cpu_hidden[i];
        // takes the value from the same cpu_output matrix (iteratively) and compute them and then replace them
        // get the value num * (var + num2 * var^3)
        float inner = 0.7978845608f * (var + 0.044715f * var*var*var);
        cpu_hidden[i] = 0.5f * var * (1.0f + tanh(inner)); // add this to the matrix back x[i]
    }
    for (int row = 0; row<m2; row++){
        for (int col = 0; col<n2; col++){
            float sum = 0.0f;
            for(int K = 0; K < k2; K++){
                sum += cpu_hidden[row*k2 + K] * h_w2[K*n2 + col];
            }
            cpu_output[row * n2 + col] = sum;
        }
    }
    // compute the max difference
    for (int i=0; i<seqlen*hiddenDim; i++){
        max_diff = fmaxf(max_diff, fabsf(h_output[i] - cpu_output[i]));
    }
    printf("So the final GPU compute we got is :\n %f", (double)h_output[0]);
    printf("\nSo the max difference between the GPU result and the CPU result is : \n %f", max_diff);

    cudaFree(d_w1); cudaFree(d_w2);  cudaFree(d_hidden); cudaFree(d_output);
    delete[] h_w1; delete[] h_w2; delete[] h_hidden; delete[] h_x; delete[] h_output; delete[] cpu_output; delete[] cpu_hidden;

}