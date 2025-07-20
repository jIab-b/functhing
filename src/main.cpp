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
    bool enableMouseLook = false;
    float moveSpeed = 10.0f;
    float mouseSensitivity = 0.1f;
    float* heightmap = new float[CHUNK * CHUNK];
    // Generate initial heightmap before the render loop
    generateHeightmap(heightmap, CHUNK, CHUNK, static_cast<uint32_t>(GetTime() * 1000));

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_Q)) {
            enableMouseLook = !enableMouseLook;
            if (enableMouseLook) {
                DisableCursor();
            } else {
                EnableCursor();
            }
        }
        if (IsKeyPressed(KEY_SPACE)) {
            generateHeightmap(heightmap, CHUNK, CHUNK, static_cast<uint32_t>(GetTime() * 1000));
        }
        renderer.UpdateMesh(heightmap, CHUNK, CHUNK);
        float dt = GetFrameTime();
        if (enableMouseLook) {
            Vector3 movement = { 0.0f, 0.0f, 0.0f };
            Vector3 rotation = { 0.0f, 0.0f, 0.0f };
            float zoom = 0.0f;
            float speed = IsKeyDown(KEY_LEFT_SHIFT) ? moveSpeed * 10.0f : moveSpeed;
            speed *= dt;
            if (IsKeyDown(KEY_W)) movement.z = speed;
            if (IsKeyDown(KEY_S)) movement.z = -speed;
            if (IsKeyDown(KEY_A)) movement.x = -speed;
            if (IsKeyDown(KEY_D)) movement.x = speed;
            if (IsKeyDown(KEY_SPACE)) movement.y = speed;
            if (IsKeyDown(KEY_LEFT_CONTROL)) movement.y = -speed;
            Vector2 mouseDelta = GetMouseDelta();
            rotation.x = -mouseDelta.y * mouseSensitivity;
            rotation.y = -mouseDelta.x * mouseSensitivity;
            UpdateCameraPro(&cam, movement, rotation, zoom);
        } else {
            UpdateCamera(&cam, CAMERA_ORBITAL);
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode3D(cam);
        renderer.Draw();
        EndMode3D();
        EndDrawing();
    }

    delete[] heightmap;
    CloseWindow();
    return 0;
}
