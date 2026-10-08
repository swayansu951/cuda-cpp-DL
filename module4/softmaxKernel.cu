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


}
__global__ void matmulTiled(const float* A, const float* B, float* C, int N){
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

// softmax kernel, a algorithm where we apply the softmax concept to make the computation more faster

#define ATTN_TILE 16

// inplement tree reduction method : one block per row, each threads reads the stridded slice of the row and at the end combines all the partial value
__global__ void SoftmaxKernel(const float* Q, const float* V, const float* K, float* output, int d_k, float scales, int seqLen){
    // we assign a row which is to distinguish which row this perticular CUDA block is responsible for processing.
    int row = blockIdx.x; // this block owns Q's row, row start to finish
    // and a tId which tells which thread (its index) within the block.
    int tId = threadIdx.x; // thread tId ends up owning the output dimention 'tId'
    // A shared memory array of datatype of float within for the threads within the block for the inter-thread communication 
    extern __shared__ float sdata[];
    float* k_tile = sdata;
    float* v_tile = sdata + ATTN_TILE * d_k;
    float* scores_tile = sdata + 2*ATTN_TILE * d_k;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float output_acc = 0.0f;
    // make a numKvTile where it stores the loop trip count
    int numKvTiles = (seqLen + ATTN_TILE -1) / ATTN_TILE;

    // and a rowPtr that calculates the block's specific row's starting memory address by multiplying the sequence length of the row with the row index.
    // float* rowPtr = scores + row * seqLen;
    
    // step1 : each thread finds the max over its strided slice ( basically gets the subset of elements from the data array, list or tensor by 
    // jumping forward and backward with a fixed step size )
    // in this each thread is initialized with a local max as infinity.
    // float local_max = -INFINITY;
    // for (int col=tId; col<seqLen; col+=blockDim.x){
    //     local_max = fmaxf(local_max, rowPtr[col]);
    // }
    // sdata[tId] = local_max;
    
    __syncthreads();
    
    for (int t=0; t<numKvTiles; t++){
        // step1 determine the total workload per tile (Each tile holds TILE_SIZE rows and d_k cols)
        // loads the value in the k rows and v columns into the k_tile and v_tile 
        // we have to iterate with the total elements in the tile (tile_size * d_k)
        // get the row and col in a variable
        // copy the global index value to the row and col tiles 
        int totalElements = ATTN_TILE*d_k;
        for (int idx=tId; idx<totalElements; idx+=blockDim.x){
            int localRow = idx/d_k;
            int col = idx % d_k;

            int global_row = t*ATTN_TILE + localRow;
            int global_idx = global_row * d_k + col;

            k_tile[idx] = K[global_idx];
            v_tile[idx] = V[global_idx]; 
        }
        __syncthreads();

        //TODO : step2 calculate the dot product and sotre it in the scores_tile
        if (tId < ATTN_TILE){
            float dot_product = 0.0f;
            for (int k=0; k<d_k; ++k){
                dot_product += Q[row * d_k + k] * k_tile[tId * d_k + k];
            }
            scores_tile[tId] = scales * dot_product;
        }
        __syncthreads();

        // TODO: step3 tree reduction 
        for (int s=blockDim.x/2; s > 0; s >>=1){
            // if tId is less than the s then we will update the value in teh sdata of tId index with the max value between the current tId with the one ahead.
            if (tId < s) scores_tile[tId] = fmaxf(scores_tile[tId], scores_tile[tId+s]);
            __syncthreads();
        }
        float tile_max = scores_tile[0];

        // rowPtr[tId] = row * tileRow ;
        // if (sdata[tId] > rowPtr[tId]) exp(sdata[tId] - rowPtr[tId]); // if the local tile's max is greater than the running max then do exponent of subtration of 2 values 

        float new_max = fmaxf(running_max, tile_max);
        float rescale = exp(running_max - new_max);

        running_sum *= rescale;
        output_acc *= rescale;

        // for each element in the tile we have to store the exponential value and add to the running_sum and output_acc
        for (int j=0; j<ATTN_TILE; j++){ 
            float p = exp(scores_tile[j] - new_max);
            running_sum += p; 
            output_acc += p * v_tile[j*d_k +tId];
        }
        // one question : should every thread redundantly computes or should one thread compute and share via shared memory?
        running_max = new_max;  
        __syncthreads();

    }
    output[row * d_k + tId] = output_acc/running_sum;

    // float row_max = sdata[0];
    // // ensure that the row_max has been readen before shared memory is reused 
    // __syncthreads();
    // // we assign a var (local_sum) to store the sum of the row at the end
    // float local_sum = 0.0f;
    // // each thread iterates through the assigned column and subtract with the row_max and then computes the exponential using. then overwirtes the global memory
    // // with the exponential value.
    // for (int col = tId; col < seqLen; col += blockDim.x){
    //     float val = expf(rowPtr[col] - row_max);
    //     // overwrite in place with exp value
    //     rowPtr[col] = val;
    //     local_sum += val;
    // }
    // /// then stores each threads value in a shared memory 
    // sdata[tId] = local_sum;
    // __syncthreads();

    // float row_sum = sdata[0];
    // __syncthreads();

    // // step3 : normalize
    // // completes the softmax operation by iterating through the column and divide the exponential value with the row_sum  
    // for (int col=tId; col<seqLen; col += blockDim.x){
    //     rowPtr[col] /= row_sum;
    // }

}

int divideSqrt(float d_k){
    return 1.0f/sqrtf((float)d_k);
}

int main(){

    int N = 512;
    int seqlen = 64;
    int d_k = ATTN_TILE;
    float scales = divideSqrt((float)d_k);

    size_t qKVBytes = seqlen * d_k * sizeof(float);    
    float *h_Q = new float[seqlen*d_k], *h_K = new float[seqlen*d_k], *h_V = new float[seqlen*d_k];
    float *h_output = new float[seqlen*d_k];

    srand(42);
    for (int i=0; i<seqlen*d_k; i++){
        h_Q[i] = (float)(rand() % 10) / 10.0f; h_K[i] = (float)(rand() % 10) / 10.0f; h_V[i] = (float)(rand() % 10) / 10.0f;
    }
     
    float *d_Q, *d_K, *d_V, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_Q, qKVBytes));
    CUDA_CHECK(cudaMalloc(&d_K, qKVBytes));
    CUDA_CHECK(cudaMalloc(&d_V, qKVBytes));
    CUDA_CHECK(cudaMalloc(&d_output, qKVBytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qKVBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qKVBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qKVBytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(ATTN_TILE);
    size_t sharedMemBytes = (2 * ATTN_TILE * d_k + ATTN_TILE) * sizeof(float);
    dim3 blocks((N+15) / 16, (N+15)/16);

    // matmulTiled<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    SoftmaxKernel<<<seqlen, threadsPerBlock, sharedMemBytes>>>(d_Q, d_V, d_K , d_output, d_k, scales, seqlen);

    CUDA_CHECK(cudaGetLastError()); // catches lauch time error
    CUDA_CHECK(cudaDeviceSynchronize()); // catches runtime error
    CUDA_CHECK(cudaMemcpy(h_output, d_output, qKVBytes, cudaMemcpyDeviceToHost));

    // int check_i = 3 , check_j = 7;
    // float cpu_val = 0.0f;
    float* cpu_output = new float[seqlen*d_k];
    for(int i=0; i<seqlen; i++){
        float scores[64];
        float row_max = -INFINITY;
        
        for(int j=0; j<seqlen; j++){
            float dot = 0.0f;
            for (int k=0; k<d_k; k++){
                dot += h_Q[i*d_k + k] * h_K[j*d_k + k];
            }
            scores[j] = dot * scales;
            row_max = fmaxf(row_max, scores[j]);
        }

    
        float sum = 0.0f;
        for (int j=0; j<seqlen; j++){
            scores[j] = exp(scores[j] - row_max); sum += scores[j];
        }
        for (int k=0; k<d_k; k++){
            float acc = 0.0f;
            for (int l=0; l<seqlen; l++){
                acc += scores[l] * h_V[l*d_k + k];
            }   
            cpu_output[i * d_k + k] = acc/sum;
        }
    }   
    float diff = 0.0f;
    for (int i=0; i<seqlen*d_k; i++){
        diff = fmaxf(diff, fabsf(h_output[i] - cpu_output[i]));
    }
    printf("\n Max difference should be (must be ~0): %f", diff);
    // printf("\nGPU c[%d][%d] = %f,\n CPU exceeded = %f\n", check_i, check_j, h_C[check_i * N + check_j], cpu_val);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_output);
    delete[] h_Q; delete[] h_K; delete[] h_output; delete[] cpu_output;

}