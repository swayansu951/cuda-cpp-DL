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

#define TILE_SIZE 16

__global__ void matmulKernel(const float* A, float* B, float* C, int N){
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
    
    // if ((row >= TILE_SIZE) || (col >= TILE_SIZE)){
    //     printf("Exced the limit! row : %d  |  col : %d", row, col);
    //     return;
    // }

    float sum = 0.0f;
    // so instead of using the limit as N we will now  we the (N + TILE_SIZE - 1)/ TILE_SIZE
    for (int t=0; t<(N + TILE_SIZE - 1)/TILE_SIZE; t++){
        // Now each thread load one element into shared memory
        int tileCol = t * TILE_SIZE + threadIdx.x;
        int tileRow = t * TILE_SIZE + threadIdx.y;

        // Now we have to check whether the row and column is not exceed the global matrix. This prevent when the matrix size is 
        // not evenly divided by the block size.
        // if not exceed (True) then we will load the value from the global memory arrya 'A' using 1D row-major indexing (row * N + tiledCol)
        // if not then padded by 0.0f.
        tileA[threadIdx.y][threadIdx.x] = (row < N && tileCol < N) ? A[row * N + tileCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tileRow < N && col < N) ? B[tileRow * N + col] : 0.0f;
        
        __syncthreads(); // wait until all the threads in the block finishes loading 

        for (int i = 0; i < TILE_SIZE; i++){
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        
        __syncthreads(); // wiat until all finishes done reading before the next tile overwrites
    }
    // at the end check if the row and col not exceed the N
    if (row < N && col < N){
        C[row*N + col] = sum;
    }

}

int main(){

    int N = 512;

    size_t bytes = N * N * sizeof(float);
    float *h_A = new float[N*N], *h_B = new float[N*N], *h_C = new float[N*N];

    for (int i=0; i<N*N; i++){
        h_A[i] = (float)(rand() % 10); h_B[i] = (float)(rand() % 10);
    }
     
    float *d_A, *d_B, *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16,16);
    dim3 blocks((N+15) / 16, (N+15)/16);

    matmulKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // catches lauch time error
    CUDA_CHECK(cudaDeviceSynchronize()); // catches runtime error

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    int check_i = 3 , check_j = 7;
    float cpu_val = 0.0f;
    
    for(int i=0; i<N; i++){
        cpu_val += h_A[check_i*N + i] * h_B[i*N + check_j];
    }

    printf("\nGPU c[%d][%d] = %f,\n CPU exceeded = %f\n", check_i, check_j, h_C[check_i * N + check_j], cpu_val);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;


}