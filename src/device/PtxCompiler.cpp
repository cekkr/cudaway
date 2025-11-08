#include "device/PtxCompiler.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <system_error>
#include <utility>

namespace cudaway::device {
namespace {
constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;
}  // namespace

PtxCompiler::PtxCompiler(CompilerOptions options) : options_(std::move(options)) {
    if (options_.diskCacheRoot.empty()) {
        options_.diskCacheRoot = std::filesystem::temp_directory_path() / "cudaway_ptx_cache";
    }
}

CompilationResult PtxCompiler::compile(std::string_view ptxSource) {
    CompilationResult result{};
    const auto digest = compute_digest(ptxSource);

    if (options_.enableMemoryCache) {
        if (auto cached = try_load_from_memory(digest)) {
            result.gcnBinary = std::move(*cached);
            result.cacheHit = true;
            result.cacheKey = digest;
            return result;
        }
    }

    if (options_.enableDiskCache) {
        if (auto cached = try_load_from_disk(digest)) {
            result.gcnBinary = std::move(*cached);
            result.cacheHit = true;
            result.cacheKey = digest;
            result.cacheFile = cache_file_for(digest);
            store_in_memory(digest, result.gcnBinary);
            std::cout << "[ptx-jit] Cache hit (disk) for PTX payload of " << ptxSource.size()
                      << " bytes [" << result.cacheFile.string() << "]\n";
            return result;
        }
    }

    // TODO: Integrate real PTX parser and LLVM AMDGPU backend once available.
    result.gcnBinary.resize(sizeof(digest));
    std::memcpy(result.gcnBinary.data(), &digest, sizeof(digest));
    result.cacheHit = false;
    result.cacheKey = digest;

    store_in_memory(digest, result.gcnBinary);
    store_on_disk(digest, result.gcnBinary);
    result.cacheFile = cache_file_for(digest);

    std::cout << "[ptx-jit] Compiled PTX payload of " << ptxSource.size() << " bytes "
              << "(digest=0x" << std::hex << std::setw(16) << std::setfill('0') << digest << std::dec
              << ")\n";

    return result;
}

PtxCompiler::Digest PtxCompiler::compute_digest(std::string_view ptxSource) noexcept {
    std::uint64_t hash = kFNVOffset;
    for (unsigned char c : ptxSource) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= kFNVPrime;
    }
    return hash;
}

std::filesystem::path PtxCompiler::cache_file_for(Digest digest) const {
    std::ostringstream name;
    name << std::hex << std::setw(16) << std::setfill('0') << digest << ".bin";
    return options_.diskCacheRoot / name.str();
}

std::optional<std::vector<std::uint8_t>> PtxCompiler::try_load_from_memory(Digest digest) {
    std::scoped_lock lock(cacheMutex_);
    if (auto it = memoryCache_.find(digest); it != memoryCache_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<std::vector<std::uint8_t>> PtxCompiler::try_load_from_disk(Digest digest) {
    const auto file = cache_file_for(digest);
    std::error_code ec;
    if (!std::filesystem::exists(file, ec)) {
        return std::nullopt;
    }
    std::ifstream input(file, std::ios::binary);
    if (!input) {
        return std::nullopt;
    }
    std::vector<std::uint8_t> blob{std::istreambuf_iterator<char>(input),
                                   std::istreambuf_iterator<char>()};
    return blob;
}

void PtxCompiler::store_in_memory(Digest digest, const std::vector<std::uint8_t>& blob) {
    if (!options_.enableMemoryCache) {
        return;
    }
    std::scoped_lock lock(cacheMutex_);
    memoryCache_[digest] = blob;
}

void PtxCompiler::store_on_disk(Digest digest, const std::vector<std::uint8_t>& blob) {
    if (!options_.enableDiskCache) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(options_.diskCacheRoot, ec);
    const auto file = cache_file_for(digest);
    std::ofstream output(file, std::ios::binary);
    if (!output) {
        std::cerr << "[ptx-jit] Warning: failed to write cache file " << file << '\n';
        return;
    }
    output.write(reinterpret_cast<const char*>(blob.data()),
                 static_cast<std::streamsize>(blob.size()));
}

}  // namespace cudaway::device
