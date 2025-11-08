#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cudaway::device {

struct CompilationResult {
    std::vector<std::uint8_t> gcnBinary;
    bool cacheHit{false};
    std::uint64_t cacheKey{0};
    std::filesystem::path cacheFile;
};

struct CompilerOptions {
    std::filesystem::path diskCacheRoot;
    bool enableDiskCache{true};
    bool enableMemoryCache{true};
};

class PtxCompiler {
   public:
    explicit PtxCompiler(CompilerOptions options = {});

    CompilationResult compile(std::string_view ptxSource);

    [[nodiscard]] const CompilerOptions& options() const noexcept { return options_; }

   private:
    using Digest = std::uint64_t;

    static Digest compute_digest(std::string_view ptxSource) noexcept;
    [[nodiscard]] std::filesystem::path cache_file_for(Digest digest) const;
    std::optional<std::vector<std::uint8_t>> try_load_from_memory(Digest digest);
    std::optional<std::vector<std::uint8_t>> try_load_from_disk(Digest digest);
    void store_in_memory(Digest digest, const std::vector<std::uint8_t>& blob);
    void store_on_disk(Digest digest, const std::vector<std::uint8_t>& blob);

    CompilerOptions options_;
    std::mutex cacheMutex_;
    std::unordered_map<Digest, std::vector<std::uint8_t>> memoryCache_;
};

}  // namespace cudaway::device
