import os
import subprocess
import re

def generate_import_lib():
    dll_path = r'lib\raylib.dll'
    lib_path = r'lib\raylib.lib'
    if os.path.exists(lib_path):
        print("Import library already exists.")
        return
    print("Generating import library from DLL...")

    exports_file = 'raylib_exports.txt'
    def_file = 'raylib.def'

    # Run dumpbin to get exports
    try:
        subprocess.check_call(f'dumpbin /exports "{dll_path}" > "{exports_file}"', shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dumpbin: {e}")
        raise

    # Parse the exports
    exports = []
    with open(exports_file, 'r') as f:
        lines = f.readlines()
        started = False
        for line in lines:
            line = line.strip()
            if 'ordinal hint RVA      name' in line:
                started = True
                continue
            if started and line and not line.startswith('Summary'):
                parts = re.split(r'\s+', line)
                if len(parts) >= 4:
                    ordinal = parts[0]
                    name = parts[3]
                    exports.append((name, ordinal))

    if not exports:
        print("No exports found in DLL.")
        os.remove(exports_file)
        raise ValueError("No exports found")

    # Create .def file
    with open(def_file, 'w') as f:
        f.write('LIBRARY "raylib.dll"\n')
        f.write('EXPORTS\n')
        for name, ordinal in exports:
            f.write(f'    {name} @{ordinal}\n')

    # Run lib to create .lib
    try:
        subprocess.check_call(f'lib /def:"{def_file}" /out:"{lib_path}" /machine:x64', shell=True)
        print("Import library generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running lib: {e}")
        raise

    # Cleanup
    os.remove(exports_file)
    os.remove(def_file)

if __name__ == "__main__":
    print("Starting compilation process. Ensure this is run in a Developer Command Prompt for Visual Studio.")
    try:
        generate_import_lib()
    except Exception as e:
        print(f"Failed to generate import library: {e}")
        exit(1)

    print("Compiling the project with nvcc...")
    compile_cmd = (
        'nvcc -o functhing.exe '
        'src\\main.cpp src\\Renderer.cpp src\\Utils.cpp src\\TerrainGenerator.cu '
        '-Iinclude -Iinclude\\glm '
        '-Xlinker /LIBPATH:lib -Xlinker raylib.lib '
        '-Xlinker opengl32.lib -Xlinker gdi32.lib -Xlinker user32.lib '
        '-Xlinker winmm.lib -Xlinker shell32.lib'
    )
    try:
        subprocess.check_call(compile_cmd, shell=True)
        print("Compilation successful. Executable: functhing.exe")
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        exit(1) 