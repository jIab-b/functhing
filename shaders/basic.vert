#version 330 core
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
uniform mat4 uMVP;
out vec3 vNormal;
void main() {
    gl_Position = uMVP * vec4(inPos, 1.0);
    vNormal = inNormal;
}
