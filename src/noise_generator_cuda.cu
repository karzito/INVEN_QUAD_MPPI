#include "noise_generator_cuda.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>

// Define constants
#define BLOCK_SIZE 256  // Number of threads per block

// CUDA error checking macro
#define CUDA_CALL(func)                                                         \
    {                                                                           \
        cudaError_t err = (func);                                               \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error in file '%s' at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

// Kernel to generate random noise
__global__ void generate_noise(double* noise_array, int num_samples, double mean, double stddev, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    // Initialize CURAND state
    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    // Generate Gaussian noise
    noise_array[idx] = mean + stddev * curand_normal_double(&state);
}

void generateNoiseCUDA(std::vector<double>& noise, double mean, double stddev) {
    int num_samples = noise.size();
    double* d_noise_array;

    // Allocate memory on the device
    CUDA_CALL(cudaMalloc((void**)&d_noise_array, num_samples * sizeof(double)));

    // Get CPU time as seed (using std::chrono)
    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Launch kernel with enough blocks to cover all samples
    int num_blocks = (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generate_noise<<<num_blocks, BLOCK_SIZE>>>(d_noise_array, num_samples, mean, stddev, seed);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Copy data back to the host
    CUDA_CALL(cudaMemcpy(noise.data(), d_noise_array, num_samples * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CALL(cudaFree(d_noise_array));
}
