#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

using namespace std;

// we will see how different loop approach affect our logic
// in this we took 2 const matrixes A,B and a matrix C with integer N 
void MatrixCompute_ijk(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N){
    // loop over the 3 matrixes the time of the interger N which represents the dimention.
    for (int i=0; i < N; i ++){
        for (int j=0; j<N; j++){
            // here initialize a float variable to store the Compute value 
            float var = 0.0f;
            // compute the dot product of row i of A and column j of B
            for (int k=0; k<N; k++){
                // this is a dot product of matrix A and B
                // row i and column k of matrix A 
                // row k and column j of matrix B
                // both rows are N times 
                var += A[i*N + k] * B[k*N + j];
            }
            // At the end it stores the result in a 1D matrix C 
            C[i*N + j] = var;
        }
    }

}

void MatrixCompute_ikj(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N){
    // loop over the 3 matrixes the time of the interger N which represents the dimention.
    std::fill(C.begin(), C.end(), 0.0f);
    for (int i=0; i < N; i ++){
        for (int k=0; k<N; k++){
            // here initialize a float variable to store the Compute value 
            float var = A[i*N + k];
            // compute the dot product of row i of A and column j of B
            for (int j=0; j<N; j++){
                // same as above function for 'ijk'
                // At the end it stores the result in a 1D matrix C 
                C[i*N + j]  += var * B[k*N + j];
            }
        }
    }

}

int main(){
    int N = 512; // then 512, 1024

    std::vector<float> A(N*N), B(N*N), C1(N*N), C2(N*N, 0.0f);

    // generate a random number with fixed seed 42
    std::mt19937 rng(42);
    // generate a random floating which is evenly distributes accross a specific range
    std::uniform_real_distribution<float> dist(0.0f,1.0f);

    // now we will loop through the matrix A and B to store the random float values 
    // and assign to it (by reference to the actual value in the container to modify directly instead of coping)
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    MatrixCompute_ijk(A, B, C1, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    MatrixCompute_ikj(A, B, C2, N);
    auto t2 = std::chrono::high_resolution_clock::now();

    // We now calculate the latency by subtracting the time gap 
    // it stores the time in double and the minimal unit time is millisec
    double ijk_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    double ikj_ms = std::chrono::duration<double, std::milli>(t2-t1).count();

    // We have to calculate the max. difference between the two looped outcomes (i.e. ijk | ikj)
    float max_diff = 0.0f;
    for (int i=0; i<N*N; i++){
        // stores the floating max value where the difference is kept as fabs (floating absolute value) so that no negative arises..
        max_diff = std::fmax(max_diff, ::fabs(C1[i] - C2[i])); 
    }

    std::cout << "N = " << N << "\n";
    std::cout << "ijk: " << ijk_ms << " ms\n";
    std::cout << "ikj: " << ikj_ms << " ms\n";
    std::cout << "speedup: " << ijk_ms / ikj_ms << "x\n";
    std::cout << "max diff (correctness check, should be ~0): " << max_diff << "\n";

}

// run this and see the difference between the 2 algos, which one performs the best.. :)