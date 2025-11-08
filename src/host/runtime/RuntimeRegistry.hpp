#pragma once

#include <cstddef>
#include <iosfwd>
#include <map>
#include <string_view>

#include "host/runtime/RuntimeStubTable.generated.hpp"

namespace cudaway::host::runtime {

struct RuntimeApiSummary {
    std::size_t totalEntries{0};
    std::size_t hipDocumented{0};
    std::size_t hipMissing{0};
    std::map<std::string_view, std::size_t, std::less<>> statusCounts;
};

[[nodiscard]] const RuntimeApiEntry* find_runtime_entry(std::string_view cudaSymbol) noexcept;
[[nodiscard]] RuntimeApiSummary summarize_runtime_table() noexcept;
void print_runtime_summary(const RuntimeApiSummary& summary, std::ostream& os);

}  // namespace cudaway::host::runtime
