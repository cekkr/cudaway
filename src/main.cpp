#include <iostream>
#include <string>

#include "host/HostApiLayer.hpp"

int main() {
    cudaway::host::HostApiLayer hostLayer;
    const auto initStatus = hostLayer.initialize();
    if (!initStatus.ok()) {
        std::cerr << "Failed to initialise CUDAway host layer\n";
        return 1;
    }

    const std::string fakePtx = R"(
        //
        // .visible .entry vector_add(...)
        //
    )";

    const auto moduleResult = hostLayer.load_module_from_ptx(fakePtx);
    if (!moduleResult.status.ok()) {
        std::cerr << "Failed to load module: " << moduleResult.status.detail << '\n';
        return 1;
    }

    const auto functionResult = hostLayer.get_function(moduleResult.handle, "vector_add");
    if (!functionResult.status.ok()) {
        std::cerr << "Failed to resolve function: " << functionResult.status.detail << '\n';
        return 1;
    }

    const auto launchStatus =
        hostLayer.launch_kernel(functionResult.handle, cudaway::host::LaunchDimensions{
                                                           .gridX = 1,
                                                           .gridY = 1,
                                                           .gridZ = 1,
                                                           .blockX = 64,
                                                           .blockY = 1,
                                                           .blockZ = 1,
                                                       });

    if (!launchStatus.ok()) {
        std::cerr << "Launch failed: " << launchStatus.detail << '\n';
        return 1;
    }

    return 0;
}
