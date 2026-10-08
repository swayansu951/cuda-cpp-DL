// reads the file (to get the content)
//signature (to get the weights of the .pt file), filename and the size of float value (n)
// create a n size of float array 
// read the file and get the content in a variable
// open the binary file in read only mode
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
        fprintf("Short content read: %s got %zu, but we expected %zu\n", filename, readContent, numFloat);
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

layerWeights layer[0];

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

// float* h_h0_ln1_gamma = loadWeight("weights/h0_ln1_gamma.bin", 384);
// float* h_h0_attn_qkv = loadWeight("weights/h0_attn_qkv.bin", 384 * 1152);
// float* h_h0_attn_proj = loadWeight("weights/h0_attn_proj.bin", 384 * 384);
// float* h_h0_mlp_fc = loadWeight("weights/h0_mlp_fc.bin", 384 * 1536);
// float* h_h0_mlp_proj = loadWeight("weights/h0_mlp_proj.bin", 1536 * 384);
// float* h_h0_ln2_gamma = loadWeight("weights/h0_ln2_gamma.bin", 384);

__global__ void embedKernel(const int* tokenIds, const float* wte, const float* wpe, float* output, int hiddenDim){
    // hiddenDim is of 386
    int token = blockIdx.x;
    int i = threadIdx.x;
    
    int tId = tokenIds[token];
    output[token * hiddenDim + i] = wte[tId * hiddenDim + i] + wpe[token * hiddenDim + i];
}