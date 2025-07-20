#include "Renderer.hpp"
#include <raylib.h>

void Renderer::Init(const char* vsPath, const char* fsPath) {
    shader = LoadShader(vsPath, fsPath);
    width = height = 0;
    vao = vbo = ibo = 0;
}

void Renderer::UpdateMesh(const float* heightmap, int w, int h) {
    width = w;
    height = h;
    // TODO: generate mesh from heightmap
}

void Renderer::Draw() {
    BeginShaderMode(shader);
    // TODO: draw mesh
    EndShaderMode();
}
