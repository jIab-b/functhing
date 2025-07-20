#include "Renderer.hpp"
#include "TerrainGenerator.hpp"
#include <raylib.h>
#include <cstdint>

int main() {
    const int CHUNK = 256;
    InitWindow(800, 600, "ProceduralTerrain");
    Renderer renderer;
    renderer.Init("shaders/basic.vert", "shaders/basic.frag");
    Camera3D cam = { {0.0f, 50.0f, 100.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 60.0f, 0 };
    UpdateCamera(&cam, CAMERA_ORBITAL);
    float* heightmap = new float[CHUNK * CHUNK];

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            generateHeightmap(heightmap, CHUNK, CHUNK, static_cast<uint32_t>(GetTime() * 1000));
        }
        renderer.UpdateMesh(heightmap, CHUNK, CHUNK);

        BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode3D(cam);
        renderer.Draw();
        EndMode3D();
        EndDrawing();

        UpdateCamera(&cam, CAMERA_ORBITAL);
    }

    delete[] heightmap;
    CloseWindow();
    return 0;
}
