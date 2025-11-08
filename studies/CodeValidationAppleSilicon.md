### What’s possible to compile on Apple Silicon macOS

* **Native toolchains**

  * **CUDA:** NVIDIA doesn’t ship `nvcc` or CUDA libraries for macOS/ARM. You can’t truly compile `.cu` (with `__global__`, etc.) on Apple Silicon without a Linux toolchain.
  * **ROCm/HIP:** ROCm isn’t supported on macOS. There’s no official `hipcc` for macOS/ARM.

* **Workable approach (compile-only validation)**
  Use **x86_64 Linux toolchains inside Docker** on your M-series Mac. You won’t be able to run the binaries, but you can fully **compile and link** (or at least compile to objects/ptx) to validate your source code and catch real compiler errors.

---

## Option A — CUDA compile in an x86_64 Linux container

1. Install Docker Desktop (Apple Silicon supports emulating x86_64 containers via qemu).
2. Run an x86_64 CUDA “devel” image and mount your project:

```bash
# From your project root
docker run --rm -it \
  --platform=linux/amd64 \
  -v "$PWD":/ws -w /ws \
  nvidia/cuda:12.6.0-devel-ubuntu22.04 \
  bash
```

3. Inside the container, compile:

```bash
# Single TU
nvcc -std=c++17 -O2 -c src/my_kernels.cu -o build/my_kernels.o

# If you want PTX only (no link step)
nvcc -std=c++17 -O2 -ptx src/my_kernels.cu -o build/my_kernels.ptx

# With CMake (example)
apt-get update && apt-get install -y cmake ninja-build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -v
```

> This gives you real CUDA diagnostics against the actual headers/device libs. No GPU needed for compilation.

---

## Option B — HIP/ROCm compile (HIP AMD backend) in an x86_64 Linux container

ROCm provides `hipcc` on Linux/x86_64. Again, you can compile (and even link) without a GPU.

```bash
docker run --rm -it \
  --platform=linux/amd64 \
  -v "$PWD":/ws -w /ws \
  rocm/dev-ubuntu-22.04:5.7 \
  bash
```

Inside:

```bash
# Example HIP compilation
hipcc -std=c++17 -O2 -c src/my_hip_kernels.hip -o build/my_hip_kernels.o

# Target a specific arch if you want device codegen (still not executed)
hipcc --offload-arch=gfx1100 -c src/my_hip_kernels.hip -o build/my_hip_kernels.o
```

> You can also stop at object/LLVM bitcode stages if you just want front-end validation.

---

## Option C — HIP with CUDA backend (useful if you only want one source)

HIP can target CUDA via the “HIP-on-CUDA” path. That lets you validate HIP code using NVIDIA’s CUDA toolchain—still in Linux/x86_64:

```bash
# In an amd64 Linux container with HIP installed and CUDA available:
export HIP_PLATFORM=nvidia
export CUDA_PATH=/usr/local/cuda
hipcc -std=c++17 -O2 -c src/my_hip_code.hip -o build/hip_on_cuda.o
```

This is handy if your common code is HIP-first and you want to catch errors without an AMD stack.

---

## Option D — “Header-only” syntax checks on macOS (limited)

If you absolutely must compile *something* on macOS without Docker:

* You can sometimes get away with **host-only** translation units that include CUDA/ROCm headers and don’t use CUDA language extensions (no `__global__`, no device code).
* Compile with Clang as ordinary C++ and **don’t link**:

```bash
clang++ -std=c++20 -I/path/to/cuda/include -I/path/to/hip/include \
  -fsyntax-only src/host_only_wrapper.cpp
```

But the moment you need CUDA/HIP language constructs (kernels, attributes, builtins), you’ll hit a wall without the proper toolchains. This is only good for basic interface checks.

---

## Suggested CMake skeleton (multi-backend, container-friendly)

```cmake
cmake_minimum_required(VERSION 3.24)
project(MultiGPU LANGUAGES CXX)

option(ENABLE_CUDA "Build CUDA backend" ON)
option(ENABLE_HIP  "Build HIP backend" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ENABLE_CUDA)
  enable_language(CUDA)
  add_library(cuda_backend OBJECT src/cuda/my_kernels.cu)
  target_compile_features(cuda_backend PRIVATE cxx_std_17)
  # Example flags:
  target_compile_options(cuda_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcudafe --display_error_number>)
endif()

if(ENABLE_HIP)
  # Expect hipcc toolchain inside Linux container
  find_package(HIP REQUIRED)
  hip_add_library(hip_backend OBJECT src/hip/my_hip_kernels.hip)
  target_compile_features(hip_backend PRIVATE cxx_std_17)
  # Example arch (only for device codegen):
  # target_compile_options(hip_backend PRIVATE --offload-arch=gfx1100)
endif()

add_library(core STATIC src/common/core.cpp $<TARGET_OBJECTS:cuda_backend> $<TARGET_OBJECTS:hip_backend>)
```

Then in your **Docker** runs for each backend:

```bash
# CUDA build
docker run --rm -it --platform=linux/amd64 -v "$PWD":/ws -w /ws \
  nvidia/cuda:12.6.0-devel-ubuntu22.04 \
  bash -lc 'apt-get update && apt-get install -y cmake ninja-build && cmake -S . -B build -G Ninja -DENABLE_CUDA=ON -DENABLE_HIP=OFF && cmake --build build -v'

# HIP build
docker run --rm -it --platform=linux/amd64 -v "$PWD":/ws -w /ws \
  rocm/dev-ubuntu-22.04:5.7 \
  bash -lc 'apt-get update && apt-get install -y cmake ninja-build && cmake -S . -B build -G Ninja -DENABLE_CUDA=OFF -DENABLE_HIP=ON && cmake --build build -v'
```

---

## Bottom line

* **Directly on macOS Apple Silicon:** you can’t properly compile CUDA/HIP device code.
* **For real compile-time validation:** run the **Linux/x86_64 CUDA and ROCm toolchains inside Docker** on your Mac. This gives you accurate compiler errors for both APIs without needing to execute anything.
