#include <iostream>
#include <vector>

// this is a example of pass-by-value and pass-by-refernce and 
// which matrix multiplication is faster (row major[faster] or column major[slower]) and
// AOS/SOA (array of structures/ structures of array)
int main(){
    // void process(const std::vector<float>& data) {....} // This means no copy, Can't modify
    // void process(std::vector<float>& data) {....} // This means no copy, Can modify
    // void process(std::vector<float> data) {....} // This means, copies the whole vector - usually wrong

    // We will see how matrix multiplication and addtion is done
    // int W = 5;
    // int H = 5;
    // int row = 2;
    // int col = 4;
    // float* matrix = new float[W * H];
    // int sum = 0;
    // // float val = matrix[row * W + col]; // in this we explicitely type casted to int from float
    // // // else we could also set the variables to int either..
    // // std::cout << val << "The matrix value is : " << *matrix;
    
    // // we will use loop to calculate the matrix arithmetic..
    // // This will be a faster arithmetic : walks memory in order it's actually laid out
    // sum = 0;
    // for (int row = 0; row < H; row++){
    //     for (int col = 0; col < W; col++){
    //         sum += matrix[row * W + col];
    //     }
    // }
    
    // std::cout << "This is faster comutation : " << sum << " ";

    // // This is slow : jumps around the memory 
    // for (int col = 0; col < W; col++){
    //     for (int row = 0; row < H; row++){
    //         sum += matrix[row * W + col];
    //     }
    // }
    // std::cout << "This is slower comutation : " << sum;

    // delete[] matrix;

    // To test AoS and SoA
    //AoS, if you want to process a single specific point entierlt at a time (eg. movin a single player)
    {
        float x, y, z;
    };
    Point points[1000];
    std::cout << "The AoS is used " ;
    
    // SoA
    // mass parallel operation, when you want to compute only x or y that will make the CPU to secuentially update cache without wastin space 
    // loading unwanted values..
    struct Points
    {
        float x[1000], y[1000], z[1000];
    };
    
    // instead of manual writting Buffer(as we did before) we can modern CPP function for much clearner managing dynamic memory.
    // Instead of raw new[]/delete[] via RAII we can use the std::verctor<> that helps in automatic cleanup 
    std::vector<float> data(1000, 0.0f);
    data[42] = 3.14f;
    std::cout << "The data : " << data[42];
}