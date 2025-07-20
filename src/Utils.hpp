#pragma once

// Simple smoothstep implementation
inline float smoothstep(float edge0, float edge1, float x) {
    x = (x - edge0) / (edge1 - edge0);
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    return x * x * (3 - 2 * x);
}

// Blend factor for overlap seams
inline float blendFactor(float t) {
    return smoothstep(0.0f, 1.0f, t);
}
