#include "host/runtime/RuntimeRegistry.hpp"

#include <algorithm>
#include <iostream>
#include <ostream>

namespace cudaway::host::runtime {

const RuntimeApiEntry* find_runtime_entry(std::string_view cudaSymbol) noexcept {
    const auto predicate = [cudaSymbol](const RuntimeApiEntry& entry) {
        return entry.cudaSymbol == cudaSymbol;
    };
    const auto it = std::find_if(kRuntimeApiTable.begin(), kRuntimeApiTable.end(), predicate);
    if (it == kRuntimeApiTable.end()) {
        return nullptr;
    }
    return &(*it);
}

RuntimeApiSummary summarize_runtime_table() noexcept {
    RuntimeApiSummary summary{};
    summary.totalEntries = kRuntimeApiTable.size();
    for (const auto& entry : kRuntimeApiTable) {
        if (entry.hipDocumented) {
            summary.hipDocumented += 1;
        } else {
            summary.hipMissing += 1;
        }
        summary.statusCounts[entry.status] += 1;
    }
    return summary;
}

void print_runtime_summary(const RuntimeApiSummary& summary, std::ostream& os) {
    os << "[runtime-stubs] total=" << summary.totalEntries
       << " hip-doc=" << summary.hipDocumented << " hip-missing=" << summary.hipMissing << '\n';
    for (const auto& [status, count] : summary.statusCounts) {
        os << "[runtime-stubs]   status=" << status << " count=" << count << '\n';
    }
}

}  // namespace cudaway::host::runtime

