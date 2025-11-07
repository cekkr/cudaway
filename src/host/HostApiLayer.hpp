#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "common/HandleTable.hpp"
#include "common/Types.hpp"
#include "device/PtxCompiler.hpp"

namespace cudaway::host {

struct LaunchDimensions {
    std::uint32_t gridX{1};
    std::uint32_t gridY{1};
    std::uint32_t gridZ{1};
    std::uint32_t blockX{1};
    std::uint32_t blockY{1};
    std::uint32_t blockZ{1};
};

class HostApiLayer {
   public:
    bool initialize();

    types::ModuleHandle load_module_from_ptx(std::string ptxSource);
    types::FunctionHandle get_function(types::ModuleHandle module, std::string_view kernelName);
    void launch_kernel(types::FunctionHandle function, LaunchDimensions dims);

   private:
    struct HipModule {
        std::string debugName;
        std::vector<std::uint8_t> binary;
    };

    struct HipFunction {
        std::string kernelName;
        types::ModuleHandle module;
    };

    device::PtxCompiler compiler_;
    common::HandleTable<types::ModuleHandle, HipModule> modules_;
    common::HandleTable<types::FunctionHandle, HipFunction> functions_;
};

}  // namespace cudaway::host
