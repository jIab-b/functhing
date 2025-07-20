#version 330 core
in vec3 vNormal;
out vec4 fragColor;
void main() {
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(n, lightDir), 0.0);
    vec3 baseColor = vec3(0.2, 0.7, 0.3);
    fragColor = vec4(baseColor * diff + 0.1, 1.0);
}
