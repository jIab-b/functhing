#include "TerrainGenerator.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Add device-side hash, lerp, and valueNoise2D functions
__device__ unsigned int hashUint(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__ float hash21(unsigned int x, unsigned int y, unsigned int seed) {
    unsigned int h = hashUint(x + seed) ^ hashUint(y + seed);
    return (float)h * (1.0f / 4294967295.0f);
}

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float valueNoise2D(float x, float y, unsigned int seed) {
    int x0 = floorf(x);
    int y0 = floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float sx = x - x0;
    float sy = y - y0;
    float n00 = hash21(x0, y0, seed);
    float n10 = hash21(x1, y0, seed);
    float n01 = hash21(x0, y1, seed);
    float n11 = hash21(x1, y1, seed);
    float ix0 = lerp(n00, n10, sx);
    float ix1 = lerp(n01, n11, sx);
    return lerp(ix0, ix1, sy);
}

__global__ void generateNoiseKernel(float* out, int w, int h, uint32_t seed) {
    // Implemented multi-octave value noise
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    float fx = (float)x;
    float fy = (float)y;
    const int OCTAVES = 4;
    const float BASE_FREQ = 1.0f / 64.0f;
    const float BASE_AMP = 1.0f;
    float hVal = 0.0f;
    float freq = BASE_FREQ;
    float amp = BASE_AMP;
    for (int i = 0; i < OCTAVES; ++i) {
        float n = valueNoise2D(fx * freq, fy * freq, seed);
        hVal += n * amp;
        freq *= 2.0f;
        amp *= 0.5f;
    }
    out[idx] = hVal;
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
