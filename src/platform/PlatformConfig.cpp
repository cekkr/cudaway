#include "platform/PlatformConfig.hpp"

#include <array>
#include <cstdlib>
#include <system_error>

namespace cudaway::platform {
namespace {

std::filesystem::path read_env_path(const char* name) {
    if (const char* raw = std::getenv(name); raw && *raw) {
        return std::filesystem::path(raw);
    }
    return {};
}

std::filesystem::path default_rocm_root() {
#if defined(_WIN32)
    return std::filesystem::path("C:/Program Files/AMD/ROCm");
#else
    return std::filesystem::path("/opt/rocm");
#endif
}

}  // namespace

PlatformConfig detect_platform() {
    PlatformConfig config{};
#if defined(_WIN32)
    config.hostLabel = "windows";
    config.supportsPreload = false;
    config.needsDllProxy = true;
    config.workaroundHint =
        "Install the HIP SDK and set CUDAWAY_ROCM_WINDOWS_ROOT before running the proxy DLLs.";
    constexpr std::array<const char*, 3> envKeys = {"CUDAWAY_ROCM_WINDOWS_ROOT", "ROCM_PATH", "HIP_PATH"};
#else
    config.hostLabel = "linux";
    config.supportsPreload = true;
    config.needsDllProxy = false;
    config.workaroundHint =
        "Use LD_PRELOAD with the CUDAway shim and ensure /opt/rocm (or ROCM_PATH) is installed.";
    constexpr std::array<const char*, 2> envKeys = {"ROCM_PATH", "HIP_PATH"};
#endif

    for (const auto* key : envKeys) {
        auto candidate = read_env_path(key);
        if (!candidate.empty()) {
            config.hipRuntimeRoot = std::move(candidate);
            break;
        }
    }

    if (config.hipRuntimeRoot.empty()) {
        config.hipRuntimeRoot = default_rocm_root();
    }

    std::error_code ec;
    config.hasHipRuntime = std::filesystem::exists(config.hipRuntimeRoot, ec);

#if defined(_WIN32)
    config.librarySearchPaths = {config.hipRuntimeRoot / "bin", config.hipRuntimeRoot / "lib"};
#else
    config.librarySearchPaths = {config.hipRuntimeRoot / "lib", config.hipRuntimeRoot / "lib64"};
#endif

    return config;
}

}  // namespace cudaway::platform
