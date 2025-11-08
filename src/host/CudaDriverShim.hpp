#pragma once

#include <mutex>
#include <string>
#include <string_view>

#include "host/CudaDriverTypes.hpp"
#include "host/HostApiLayer.hpp"

namespace cudaway::host {

class CudaDriverShim {
   public:
    static CudaDriverShim& instance();

    CUresult cuInit(unsigned int flags);
    CUresult cuDeviceGetCount(int* count);
    CUresult cuDeviceGet(CUdevice* device, int ordinal);
    CUresult cuDeviceGetName(char* name, int length, CUdevice device);
    CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice device);
    CUresult cuDevicePrimaryCtxRelease(CUdevice device);
    CUresult cuCtxSetCurrent(CUcontext ctx);
    CUresult cuModuleLoadData(CUmodule* module, const void* image);
    CUresult cuModuleGetFunction(CUfunction* function, CUmodule module, const char* kernelName);
    CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                            unsigned int gridDimZ, unsigned int blockDimX,
                            unsigned int blockDimY, unsigned int blockDimZ,
                            unsigned int sharedMemBytes, CUstream hStream, void** kernelParams,
                            void** extra);

   private:
    CudaDriverShim() = default;

    HostApiLayer::DriverStatus ensure_initialized_locked();
    static CUresult map_status(const HostApiLayer::DriverStatus& status);
    static std::string_view stub_device_name() noexcept;

    std::mutex mutex_;
    HostApiLayer host_;
    bool initialized_{false};
    CUdevice primaryDevice_{0};
    types::ContextHandle primaryContext_{0};
};

}  // namespace cudaway::host

extern "C" {

CUresult cuInit(unsigned int flags);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetName(char* name, int length, CUdevice device);
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice device);
CUresult cuDevicePrimaryCtxRelease(CUdevice device);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleGetFunction(CUfunction* function, CUmodule module, const char* kernelName);
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra);

}
