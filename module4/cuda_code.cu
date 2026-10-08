// __global__ void kernel(...)
// __device__ float helper(...)
// __host__ void normal(...)

int i = blockIdx.x * blockDim.x + threadIdx.x 
// Here threadIdx.x :- what's the thread position in the block
// blockDim.x :- how many threads are there in a block
// blockIdx.x :- which block this thread belongs to
