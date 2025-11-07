#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cudaway::device {

struct CompilationResult {
    std::vector<std::uint8_t> gcnBinary;
    bool cacheHit{false};
};

class PtxCompiler {
   public:
    CompilationResult compile(std::string_view ptxSource);
};

}  // namespace cudaway::device
