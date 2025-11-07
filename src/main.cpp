#include <iostream>
#include <string>

#include "host/HostApiLayer.hpp"

int main() {
    cudaway::host::HostApiLayer hostLayer;
    if (!hostLayer.initialize()) {
        std::cerr << "Failed to initialise CUDAway host layer\n";
        return 1;
    }

    const std::string fakePtx = R"(
        //
        // .visible .entry vector_add(...)
        //
    )";

    const auto module = hostLayer.load_module_from_ptx(fakePtx);
    const auto function = hostLayer.get_function(module, "vector_add");
    hostLayer.launch_kernel(function, cudaway::host::LaunchDimensions{
                                          .gridX = 1,
                                          .gridY = 1,
                                          .gridZ = 1,
                                          .blockX = 64,
                                          .blockY = 1,
                                          .blockZ = 1,
                                      });
    return 0;
}
