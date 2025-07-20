#include "Renderer.hpp"
#include <raylib.h>
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
#include "rlgl.h"

void Renderer::Init(const char* vsPath, const char* fsPath) {
    shader = LoadShader(vsPath, fsPath);
    // Initialize mesh buffers
    width = height = 0;
    indexCount = 0;
    vao = rlLoadVertexArray();
    vbo = rlLoadVertexBuffer(NULL, 0, true);
    ibo = rlLoadVertexBufferElement(NULL, 0, true);
}

void Renderer::UpdateMesh(const float* heightmap, int w, int h) {
    width = w;
    height = h;
    // Build vertex and index arrays
    int vertCount = width * height;
    int idxCount = (width - 1) * (height - 1) * 6;
    indexCount = idxCount;
    std::vector<float> data;
    data.reserve(vertCount * 6);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            float px = (float)x;
            float py = heightmap[i];
            float pz = (float)y;
            // Compute normal via central differences
            float hl = heightmap[y * width + std::max(x - 1, 0)];
            float hr = heightmap[y * width + std::min(x + 1, width - 1)];
            float hd = heightmap[std::max(y - 1, 0) * width + x];
            float hu = heightmap[std::min(y + 1, height - 1) * width + x];
            glm::vec3 n = glm::normalize(glm::vec3(hl - hr, 2.0f, hd - hu));
            data.push_back(px);
            data.push_back(py);
            data.push_back(pz);
            data.push_back(n.x);
            data.push_back(n.y);
            data.push_back(n.z);
        }
    }
    std::vector<unsigned int> inds;
    inds.reserve(idxCount);
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            unsigned int tl = y * width + x;
            unsigned int tr = tl + 1;
            unsigned int bl = (y + 1) * width + x;
            unsigned int br = bl + 1;
            // Triangle 1
            inds.push_back(tl);
            inds.push_back(bl);
            inds.push_back(tr);
            // Triangle 2
            inds.push_back(tr);
            inds.push_back(bl);
            inds.push_back(br);
        }
    }
    // Re-create GPU buffers
    rlUnloadVertexArray(vao);
    rlUnloadVertexBuffer(vbo);
    rlUnloadVertexBuffer(ibo);
    vao = rlLoadVertexArray();
    vbo = rlLoadVertexBuffer(data.data(), data.size() * sizeof(float), false);
    ibo = rlLoadVertexBufferElement(inds.data(), inds.size() * sizeof(unsigned int), false);
    // Setup vertex attributes
    rlEnableVertexArray(vao);
    rlEnableVertexBuffer(vbo);
    rlEnableVertexAttribute(0);
    rlSetVertexAttribute(0, 3, RL_FLOAT, false, 6 * sizeof(float), 0);
    rlEnableVertexAttribute(1);
    rlSetVertexAttribute(1, 3, RL_FLOAT, false, 6 * sizeof(float), 3 * sizeof(float));
    rlEnableVertexBufferElement(ibo);
    rlDisableVertexArray();
}

void Renderer::Draw() {
    BeginShaderMode(shader);
    // Draw the terrain mesh
    rlEnableVertexArray(vao);
    rlDrawVertexArrayElements(0, indexCount, 0);
    rlDisableVertexArray();
    EndShaderMode();
}
