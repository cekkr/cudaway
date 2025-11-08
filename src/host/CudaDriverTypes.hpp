#pragma once

#include <cstdint>

// Temporary CUDA driver type definitions scoped to CUDAway while we bootstrap the shim.
// The values mirror a tiny subset of CUDA's official enums so the exported symbols have
// stable signatures even before we vendor the canonical headers.
#ifndef CUDAWAY_CUDA_DRIVER_TYPES_DEFINED
#define CUDAWAY_CUDA_DRIVER_TYPES_DEFINED

enum CUresult : unsigned int {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_INVALID_MODULE = 221,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999,
};

using CUdevice = int;
using CUcontext = std::uint64_t;
using CUmodule = std::uint64_t;
using CUfunction = std::uint64_t;
using CUstream = std::uint64_t;

#endif  // CUDAWAY_CUDA_DRIVER_TYPES_DEFINED
