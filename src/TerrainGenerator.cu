#include "TerrainGenerator.hpp"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void generateNoiseKernel(float* out, int w, int h, uint32_t seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    out[idx] = 0.0f; // TODO: implement noise
}

void generateHeightmap(float* out_host, int w, int h, uint32_t seed) {
    size_t sz = w * h * sizeof(float);
    float* out_dev = nullptr;
    cudaError_t err = cudaMalloc(&out_dev, sz);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    dim3 threads(16, 16);
    generateNoiseKernel<<<blocks, threads>>>(out_dev, w, h, seed);
    cudaDeviceSynchronize();
    cudaMemcpy(out_host, out_dev, sz, cudaMemcpyDeviceToHost);
    cudaFree(out_dev);
}
