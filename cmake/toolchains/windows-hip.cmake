# Toolchain file for configuring CMake with the AMD HIP SDK on Windows.
# Usage:
#   cmake -S . -B build -G Ninja \\
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-hip.cmake \\
#     -DCUDAWAY_ROCM_WINDOWS_ROOT="C:/Program Files/AMD/ROCm"

cmake_minimum_required(VERSION 3.20)

set(_cudaway_default_root "C:/Program Files/AMD/ROCm")
if(DEFINED CUDAWAY_ROCM_WINDOWS_ROOT)
    set(_cudaway_root "${CUDAWAY_ROCM_WINDOWS_ROOT}")
elseif(DEFINED ENV{CUDAWAY_ROCM_WINDOWS_ROOT})
    set(_cudaway_root "$ENV{CUDAWAY_ROCM_WINDOWS_ROOT}")
elseif(DEFINED ENV{ROCM_PATH})
    set(_cudaway_root "$ENV{ROCM_PATH}")
else()
    set(_cudaway_root "${_cudaway_default_root}")
endif()
file(TO_CMAKE_PATH "${_cudaway_root}" _cudaway_hip_root)

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER "${_cudaway_hip_root}/bin/clang.exe")
set(CMAKE_CXX_COMPILER "${_cudaway_hip_root}/bin/clang++.exe")
set(CMAKE_RC_COMPILER "${_cudaway_hip_root}/bin/llvm-rc.exe")
set(CMAKE_AR "${_cudaway_hip_root}/bin/llvm-ar.exe")
set(CMAKE_RANLIB "${_cudaway_hip_root}/bin/llvm-ranlib.exe")

set(CMAKE_FIND_ROOT_PATH "${_cudaway_hip_root}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CUDAWAY_ROCM_WINDOWS_ROOT "${_cudaway_hip_root}" CACHE PATH "Pinned HIP SDK root" FORCE)
set(ENV{HIP_PATH} "${_cudaway_hip_root}")
