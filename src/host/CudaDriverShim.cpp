#include "host/CudaDriverShim.hpp"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

namespace cudaway::host {

CudaDriverShim& CudaDriverShim::instance() {
    static CudaDriverShim shim;
    return shim;
}

HostApiLayer::DriverStatus CudaDriverShim::ensure_initialized_locked() {
    if (initialized_) {
        return HostApiLayer::DriverStatus::success();
    }

    auto status = host_.initialize();
    if (!status.ok()) {
        return status;
    }

    types::ContextHandle ctx{};
    status = host_.get_or_create_primary_context(ctx);
    if (!status.ok()) {
        return status;
    }

    primaryContext_ = ctx;
    primaryDevice_ = 0;
    initialized_ = true;
    return HostApiLayer::DriverStatus::success();
}

CUresult CudaDriverShim::cuInit(unsigned int /*flags*/) {
    std::scoped_lock lock(mutex_);
    auto status = ensure_initialized_locked();
    return map_status(status);
}

CUresult CudaDriverShim::cuDeviceGetCount(int* count) {
    if (!count) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::scoped_lock lock(mutex_);
    auto status = ensure_initialized_locked();
    if (!status.ok()) {
        return map_status(status);
    }

    *count = 1;
    return CUDA_SUCCESS;
}

CUresult CudaDriverShim::cuDeviceGet(CUdevice* device, int ordinal) {
    if (!device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(mutex_);
    auto status = ensure_initialized_locked();
    if (!status.ok()) {
        return map_status(status);
    }
    if (ordinal != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = primaryDevice_;
    return CUDA_SUCCESS;
}

CUresult CudaDriverShim::cuDeviceGetName(char* name, int length, CUdevice device) {
    if (!name || length <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::scoped_lock lock(mutex_);
    auto status = ensure_initialized_locked();
    if (!status.ok()) {
        return map_status(status);
    }

    if (device != primaryDevice_) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const auto label = stub_device_name();
    std::strncpy(name, label.data(), static_cast<std::size_t>(length) - 1);
    name[length - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult CudaDriverShim::cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice device) {
    if (!pctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(mutex_);
    auto status = ensure_initialized_locked();
    if (!status.ok()) {
        return map_status(status);
    }
    if (device != primaryDevice_) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *pctx = primaryContext_;
    auto ctxStatus = host_.set_current_context(primaryContext_);
    return map_status(ctxStatus);
}

CUresult CudaDriverShim::cuDevicePrimaryCtxRelease(CUdevice device) {
    std::scoped_lock lock(mutex_);
    if (!initialized_) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (device != primaryDevice_) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto status = host_.release_context(primaryContext_);
    if (status.ok()) {
        initialized_ = false;
        primaryContext_ = 0;
    }
    return map_status(status);
}

CUresult CudaDriverShim::cuCtxSetCurrent(CUcontext ctx) {
    std::scoped_lock lock(mutex_);
    auto status = ensure_initialized_locked();
    if (!status.ok()) {
        return map_status(status);
    }
    auto ctxStatus = host_.set_current_context(ctx);
    return map_status(ctxStatus);
}

CUresult CudaDriverShim::cuModuleLoadData(CUmodule* module, const void* image) {
    if (!module || !image) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(mutex_);
    auto status = ensure_initialized_locked();
    if (!status.ok()) {
        return map_status(status);
    }

    // Treat the payload as PTX text for now; future revisions will pass explicit lengths and
    // binary blobs (cubin/gcn) once the JIT and loader integrate with HIP.
    const auto* asChar = static_cast<const char*>(image);
    std::string ptxPayload(asChar ? asChar : "");

    auto loadResult = host_.load_module_from_ptx(std::move(ptxPayload));
    if (!loadResult.status.ok()) {
        return map_status(loadResult.status);
    }
    *module = loadResult.handle;
    return CUDA_SUCCESS;
}

CUresult CudaDriverShim::cuModuleGetFunction(CUfunction* function, CUmodule module,
                                             const char* kernelName) {
    if (!function || !kernelName) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(mutex_);
    auto status = ensure_initialized_locked();
    if (!status.ok()) {
        return map_status(status);
    }
    auto lookup = host_.get_function(module, kernelName);
    if (!lookup.status.ok()) {
        return map_status(lookup.status);
    }
    *function = lookup.handle;
    return CUDA_SUCCESS;
}

CUresult CudaDriverShim::cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                        unsigned int gridDimZ, unsigned int blockDimX,
                                        unsigned int blockDimY, unsigned int blockDimZ,
                                        unsigned int /*sharedMemBytes*/, CUstream /*hStream*/,
                                        void** /*kernelParams*/, void** /*extra*/) {
    std::scoped_lock lock(mutex_);
    auto status = ensure_initialized_locked();
    if (!status.ok()) {
        return map_status(status);
    }
    LaunchDimensions dims{
        .gridX = gridDimX,
        .gridY = gridDimY,
        .gridZ = gridDimZ,
        .blockX = blockDimX,
        .blockY = blockDimY,
        .blockZ = blockDimZ,
    };
    auto launchStatus = host_.launch_kernel(f, dims);
    return map_status(launchStatus);
}

CUresult CudaDriverShim::map_status(const HostApiLayer::DriverStatus& status) {
    switch (status.code) {
        case HostApiLayer::DriverStatusCode::Success:
            return CUDA_SUCCESS;
        case HostApiLayer::DriverStatusCode::InvalidContext:
            return CUDA_ERROR_INVALID_CONTEXT;
        case HostApiLayer::DriverStatusCode::InvalidModule:
            return CUDA_ERROR_INVALID_MODULE;
        case HostApiLayer::DriverStatusCode::InvalidFunction:
            return CUDA_ERROR_INVALID_HANDLE;
        case HostApiLayer::DriverStatusCode::CompilationFailed:
            return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_ERROR_UNKNOWN;
}

std::string_view CudaDriverShim::stub_device_name() noexcept {
    return "cudaway::simulated_amd_gpu";
}

}  // namespace cudaway::host

extern "C" {

CUresult cuInit(unsigned int flags) {
    return cudaway::host::CudaDriverShim::instance().cuInit(flags);
}

CUresult cuDeviceGetCount(int* count) {
    return cudaway::host::CudaDriverShim::instance().cuDeviceGetCount(count);
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    return cudaway::host::CudaDriverShim::instance().cuDeviceGet(device, ordinal);
}

CUresult cuDeviceGetName(char* name, int length, CUdevice device) {
    return cudaway::host::CudaDriverShim::instance().cuDeviceGetName(name, length, device);
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice device) {
    return cudaway::host::CudaDriverShim::instance().cuDevicePrimaryCtxRetain(pctx, device);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice device) {
    return cudaway::host::CudaDriverShim::instance().cuDevicePrimaryCtxRelease(device);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    return cudaway::host::CudaDriverShim::instance().cuCtxSetCurrent(ctx);
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    return cudaway::host::CudaDriverShim::instance().cuModuleLoadData(module, image);
}

CUresult cuModuleGetFunction(CUfunction* function, CUmodule module, const char* kernelName) {
    return cudaway::host::CudaDriverShim::instance().cuModuleGetFunction(function, module,
                                                                         kernelName);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
    return cudaway::host::CudaDriverShim::instance().cuLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
        kernelParams, extra);
}

}
