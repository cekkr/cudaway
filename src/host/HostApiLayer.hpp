#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/HandleTable.hpp"
#include "common/Types.hpp"
#include "device/PtxCompiler.hpp"
#include "platform/PlatformConfig.hpp"

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
    enum class DriverStatusCode {
        Success = 0,
        InvalidContext,
        InvalidModule,
        InvalidFunction,
        CompilationFailed,
    };

    struct DriverStatus {
        DriverStatusCode code{DriverStatusCode::Success};
        std::string detail;

        [[nodiscard]] bool ok() const noexcept { return code == DriverStatusCode::Success; }

        static DriverStatus success(std::string detail = {}) {
            return DriverStatus{DriverStatusCode::Success, std::move(detail)};
        }

        static DriverStatus error(DriverStatusCode code, std::string detail) {
            return DriverStatus{code, std::move(detail)};
        }
    };

    struct ModuleLoadResult {
        DriverStatus status{};
        types::ModuleHandle handle{0};
    };

    struct FunctionLookupResult {
        DriverStatus status{};
        types::FunctionHandle handle{0};
    };

    DriverStatus initialize();

    ModuleLoadResult load_module_from_ptx(std::string ptxSource);
    FunctionLookupResult get_function(types::ModuleHandle module, std::string_view kernelName);
    DriverStatus launch_kernel(types::FunctionHandle function, LaunchDimensions dims);

    DriverStatus set_current_context(types::ContextHandle context);
    DriverStatus release_context(types::ContextHandle context);
    [[nodiscard]] std::optional<types::ContextHandle> current_context() const noexcept {
        return currentContext_;
    }

   private:
    struct HipContext {
        std::string debugName;
        bool isPrimary{false};
    };

    struct HipModule {
        std::string debugName;
        std::vector<std::uint8_t> binary;
        types::ContextHandle context;
        std::uint64_t cacheKey{0};
        bool cacheHit{false};
        std::filesystem::path cacheFile;
    };

    struct HipFunction {
        std::string kernelName;
        types::ModuleHandle module;
        types::ContextHandle context;
    };

    types::ContextHandle retain_primary_context();
    void log_context_inventory() const;
    [[nodiscard]] DriverStatus ensure_active_context() const;

    device::PtxCompiler compiler_;
    common::HandleTable<types::ContextHandle, HipContext> contexts_;
    common::HandleTable<types::ModuleHandle, HipModule> modules_;
    common::HandleTable<types::FunctionHandle, HipFunction> functions_;
    std::optional<types::ContextHandle> currentContext_;
    std::optional<types::ContextHandle> primaryContext_;
    platform::PlatformConfig platform_;
};

}  // namespace cudaway::host
