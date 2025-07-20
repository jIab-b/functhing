#pragma once

#include <cstdint>

// Generates a heightmap of size w*h into the provided buffer using the given seed
void generateHeightmap(float* out, int w, int h, uint32_t seed);
