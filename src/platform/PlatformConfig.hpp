#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace cudaway::platform {

struct PlatformConfig {
    std::string hostLabel;
    bool supportsPreload{false};
    bool needsDllProxy{false};
    bool hasHipRuntime{false};
    std::filesystem::path hipRuntimeRoot;
    std::vector<std::filesystem::path> librarySearchPaths;
    std::string workaroundHint;
};

PlatformConfig detect_platform();

}  // namespace cudaway::platform
