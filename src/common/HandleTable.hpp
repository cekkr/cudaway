#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_map>

namespace cudaway::common {

/**
 * Simple threadsafe table that generates opaque handles for native objects.
 * The handle type must be an unsigned integral so it can be auto-incremented.
 */
template <typename HandleT, typename NativeT>
class HandleTable {
   public:
    HandleT insert(NativeT native) {
        static_assert(std::is_unsigned_v<HandleT>,
                      "HandleTable assumes an unsigned integral handle type");
        const auto handle =
            static_cast<HandleT>(next_id_.fetch_add(1, std::memory_order_relaxed));
        {
            std::scoped_lock lock(mutex_);
            storage_.emplace(handle, std::move(native));
        }
        return handle;
    }

    [[nodiscard]] std::optional<NativeT> find(HandleT handle) const {
        std::scoped_lock lock(mutex_);
        if (auto it = storage_.find(handle); it != storage_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    bool erase(HandleT handle) {
        std::scoped_lock lock(mutex_);
        return storage_.erase(handle) > 0;
    }

    template <typename Fn>
    void for_each(Fn&& fn) const {
        std::scoped_lock lock(mutex_);
        for (const auto& [handle, native] : storage_) {
            fn(handle, native);
        }
    }

   private:
    mutable std::mutex mutex_;
    std::unordered_map<HandleT, NativeT> storage_;
    std::atomic_uint64_t next_id_{1};
};

}  // namespace cudaway::common
