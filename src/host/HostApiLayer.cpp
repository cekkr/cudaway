#include "host/HostApiLayer.hpp"

#include <iostream>
#include <optional>
#include <utility>

namespace cudaway::host {

bool HostApiLayer::initialize() {
    platform_ = platform::detect_platform();
    std::cout << "[host-api] Initialising CUDAway host layer (platform=" << platform_.hostLabel
              << ")\n";
    std::cout << "[host-api]   loader strategy: "
              << (platform_.supportsPreload ? "LD_PRELOAD shim" : "Proxy DLL injection") << '\n';

    std::cout << "[host-api]   HIP runtime root: " << platform_.hipRuntimeRoot.string() << '\n';
    if (!platform_.hasHipRuntime) {
        std::cout << "[host-api]   (missing on disk) " << platform_.workaroundHint << '\n';
    } else {
        for (const auto& path : platform_.librarySearchPaths) {
            std::cout << "[host-api]   search path: " << path.string() << '\n';
        }
    }

    // Later this will bootstrap HIP, contexts, and device discovery.
    return true;
}

types::ModuleHandle HostApiLayer::load_module_from_ptx(std::string ptxSource) {
    auto compilation = compiler_.compile(ptxSource);
    HipModule module{
        .debugName = "jit_module_" + std::to_string(ptxSource.size()),
        .binary = std::move(compilation.gcnBinary),
    };
    const auto handle = modules_.insert(std::move(module));
    std::cout << "[host-api] Registered module handle " << handle << '\n';
    return handle;
}

types::FunctionHandle HostApiLayer::get_function(types::ModuleHandle module,
                                                 std::string_view kernelName) {
    const auto moduleEntry = modules_.find(module);
    if (!moduleEntry) {
        throw std::runtime_error("Unknown module handle");
    }

    HipFunction func{
        .kernelName = std::string(kernelName),
        .module = module,
    };
    const auto handle = functions_.insert(std::move(func));
    std::cout << "[host-api] Created function handle " << handle << " for kernel " << kernelName
              << '\n';
    return handle;
}

void HostApiLayer::launch_kernel(types::FunctionHandle function, LaunchDimensions dims) {
    const auto func = functions_.find(function);
    if (!func) {
        throw std::runtime_error("Unknown function handle");
    }

    std::cout << "[host-api] Launching kernel '" << func->kernelName << "' "
              << "grid(" << dims.gridX << ',' << dims.gridY << ',' << dims.gridZ << ") "
              << "block(" << dims.blockX << ',' << dims.blockY << ',' << dims.blockZ << ")\n";
}

}  // namespace cudaway::host
