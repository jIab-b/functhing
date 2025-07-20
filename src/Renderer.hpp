#pragma once

#include <raylib.h>

struct Renderer {
    Shader shader;
    unsigned int vao, vbo, ibo;
    int width, height;

    // Initialize shaders and buffers
    void Init(const char* vsPath, const char* fsPath);

    // Update mesh data from heightmap
    void UpdateMesh(const float* heightmap, int w, int h);

    // Draw the current mesh
    void Draw();
};
