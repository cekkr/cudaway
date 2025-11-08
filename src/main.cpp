#include <iostream>
#include <string>

#include "host/CudaDriverShim.hpp"
#include "host/runtime/RuntimeRegistry.hpp"

int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::cerr << "cuInit failed\n";
        return 1;
    }

    int deviceCount = 0;
    if (cuDeviceGetCount(&deviceCount) != CUDA_SUCCESS || deviceCount <= 0) {
        std::cerr << "cuDeviceGetCount failed\n";
        return 1;
    }

    CUdevice device{};
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGet failed\n";
        return 1;
    }

    char deviceName[128];
    if (cuDeviceGetName(deviceName, sizeof(deviceName), device) != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGetName failed\n";
        return 1;
    }
    std::cout << "[cli] Target device: " << deviceName << '\n';

    CUcontext ctx{};
    if (cuDevicePrimaryCtxRetain(&ctx, device) != CUDA_SUCCESS) {
        std::cerr << "cuDevicePrimaryCtxRetain failed\n";
        return 1;
    }

    if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
        std::cerr << "cuCtxSetCurrent failed\n";
        return 1;
    }

    const std::string fakePtx = R"(
        //
        // .visible .entry vector_add(...)
        //
    )";

    CUmodule module{};
    if (cuModuleLoadData(&module, fakePtx.c_str()) != CUDA_SUCCESS) {
        std::cerr << "cuModuleLoadData failed\n";
        return 1;
    }

    CUfunction function{};
    if (cuModuleGetFunction(&function, module, "vector_add") != CUDA_SUCCESS) {
        std::cerr << "cuModuleGetFunction failed\n";
        return 1;
    }

    const auto launchStatus = cuLaunchKernel(function,
                                             /*gridDimX=*/1,
                                             /*gridDimY=*/1,
                                             /*gridDimZ=*/1,
                                             /*blockDimX=*/64,
                                             /*blockDimY=*/1,
                                             /*blockDimZ=*/1,
                                             /*sharedMemBytes=*/0,
                                             /*hStream=*/0,
                                             /*kernelParams=*/nullptr,
                                             /*extra=*/nullptr);
    if (launchStatus != CUDA_SUCCESS) {
        std::cerr << "cuLaunchKernel failed\n";
        return 1;
    }

    const auto runtimeSummary = cudaway::host::runtime::summarize_runtime_table();
    cudaway::host::runtime::print_runtime_summary(runtimeSummary, std::cout);

    return 0;
}
