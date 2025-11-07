#include "device/PtxCompiler.hpp"

#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>

namespace cudaway::device {

CompilationResult PtxCompiler::compile(std::string_view ptxSource) {
    CompilationResult result{};

    // TODO: Integrate real PTX parser and LLVM AMDGPU backend once available.
    const auto hash = std::hash<std::string_view>{}(ptxSource);
    result.gcnBinary.resize(sizeof(hash));
    std::memcpy(result.gcnBinary.data(), &hash, sizeof(hash));

    std::cout << "[ptx-jit] Prepared synthetic GCN binary for PTX payload of "
              << ptxSource.size() << " bytes\n";
    return result;
}

}  // namespace cudaway::device
