#include "host/HostApiLayer.hpp"

#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>

namespace cudaway::host {

HostApiLayer::DriverStatus HostApiLayer::initialize() {
    platform_ = platform::detect_platform();
    std::cout << "[host-api] Initialising CUDAway host layer (platform=" << platform_.hostLabel
              << ")\n";
    std::cout << "[host-api]   loader strategy: "
              << (platform_.supportsPreload ? "LD_PRELOAD shim" : "Proxy DLL injection") << '\n';

    std::cout << "[host-api]   HIP runtime root: " << platform_.hipRuntimeRoot.string() << '\n';
    if (!platform_.hasHipRuntime) {
        std::cout << "[host-api]   (missing on disk) " << platform_.workaroundHint << '\n';
    } else {
        for (const auto& path : platform_.librarySearchPaths) {
            std::cout << "[host-api]   search path: " << path.string() << '\n';
        }
    }

    const auto primary = retain_primary_context();
    currentContext_ = primary;
    std::cout << "[host-api]   active context: " << primary << '\n';
    log_context_inventory();

    return DriverStatus::success("Platform + primary context ready");
}

HostApiLayer::ModuleLoadResult HostApiLayer::load_module_from_ptx(std::string ptxSource) {
    ModuleLoadResult result{};
    auto status = ensure_active_context();
    if (!status.ok()) {
        result.status = status;
        return result;
    }

    auto compilation = compiler_.compile(ptxSource);
    HipModule module{
        .debugName = "jit_module_" + std::to_string(ptxSource.size()),
        .binary = std::move(compilation.gcnBinary),
        .context = *currentContext_,
        .cacheKey = compilation.cacheKey,
        .cacheHit = compilation.cacheHit,
        .cacheFile = compilation.cacheFile,
    };
    const auto handle = modules_.insert(std::move(module));
    std::ostringstream digest;
    digest << std::hex << std::setw(16) << std::setfill('0') << compilation.cacheKey;
    std::cout << "[host-api] Registered module handle " << handle << " (ctx=" << *currentContext_
              << ", cache=" << (compilation.cacheHit ? "hit" : "miss") << ", key=0x"
              << digest.str() << ")\n";
    result.handle = handle;
    result.status = DriverStatus::success(compilation.cacheHit ? "cache-hit" : "compiled");
    return result;
}

HostApiLayer::FunctionLookupResult HostApiLayer::get_function(types::ModuleHandle module,
                                                             std::string_view kernelName) {
    FunctionLookupResult result{};
    const auto moduleEntry = modules_.find(module);
    if (!moduleEntry) {
        result.status =
            DriverStatus::error(DriverStatusCode::InvalidModule, "Unknown module handle");
        return result;
    }

    if (!currentContext_ || *currentContext_ != moduleEntry->context) {
        auto ctxStatus = set_current_context(moduleEntry->context);
        if (!ctxStatus.ok()) {
            result.status = ctxStatus;
            return result;
        }
    }

    HipFunction func{
        .kernelName = std::string(kernelName),
        .module = module,
        .context = moduleEntry->context,
    };
    const auto handle = functions_.insert(std::move(func));
    std::cout << "[host-api] Created function handle " << handle << " for kernel " << kernelName
              << '\n';
    result.handle = handle;
    result.status = DriverStatus::success();
    return result;
}

HostApiLayer::DriverStatus HostApiLayer::launch_kernel(types::FunctionHandle function,
                                                       LaunchDimensions dims) {
    auto status = ensure_active_context();
    if (!status.ok()) {
        return status;
    }

    const auto func = functions_.find(function);
    if (!func) {
        return DriverStatus::error(DriverStatusCode::InvalidFunction, "Unknown function handle");
    }

    const auto moduleEntry = modules_.find(func->module);
    if (!moduleEntry) {
        return DriverStatus::error(DriverStatusCode::InvalidModule, "Unknown parent module");
    }

    if (*currentContext_ != func->context) {
        auto ctxStatus = set_current_context(func->context);
        if (!ctxStatus.ok()) {
            return ctxStatus;
        }
    }

    std::string contextLabel = "unknown";
    if (const auto ctx = contexts_.find(func->context)) {
        contextLabel = ctx->debugName;
    }

    std::cout << "[host-api] Launching kernel '" << func->kernelName << "' "
              << "grid(" << dims.gridX << ',' << dims.gridY << ',' << dims.gridZ << ") "
              << "block(" << dims.blockX << ',' << dims.blockY << ',' << dims.blockZ << ") "
              << "ctx=" << *currentContext_ << " [" << contextLabel << "]\n";
    return DriverStatus::success();
}

HostApiLayer::DriverStatus HostApiLayer::set_current_context(types::ContextHandle context) {
    const auto ctx = contexts_.find(context);
    if (!ctx) {
        return DriverStatus::error(DriverStatusCode::InvalidContext, "Unknown context handle");
    }

    currentContext_ = context;
    std::cout << "[host-api] Switched to context " << context << " (" << ctx->debugName << ")\n";
    return DriverStatus::success("context-set");
}

HostApiLayer::DriverStatus HostApiLayer::release_context(types::ContextHandle context) {
    if (!contexts_.erase(context)) {
        return DriverStatus::error(DriverStatusCode::InvalidContext,
                                   "Attempted to release unknown context");
    }

    if (currentContext_ && *currentContext_ == context) {
        currentContext_.reset();
    }

    if (primaryContext_ && *primaryContext_ == context) {
        primaryContext_.reset();
    }

    std::cout << "[host-api] Released context handle " << context << '\n';
    return DriverStatus::success("context-released");
}

HostApiLayer::DriverStatus HostApiLayer::get_or_create_primary_context(
    types::ContextHandle& out) {
    const auto handle = retain_primary_context();
    out = handle;
    return DriverStatus::success("primary-ready");
}

types::ContextHandle HostApiLayer::retain_primary_context() {
    if (primaryContext_) {
        return *primaryContext_;
    }

    HipContext ctx{
        .debugName = platform_.hostLabel + "_primary",
        .isPrimary = true,
    };

    const auto handle = contexts_.insert(std::move(ctx));
    primaryContext_ = handle;
    return handle;
}

void HostApiLayer::log_context_inventory() const {
    contexts_.for_each([](types::ContextHandle handle, const HipContext& ctx) {
        std::cout << "[host-api]   context " << handle << " name='" << ctx.debugName << "'"
                  << (ctx.isPrimary ? " [primary]" : "") << '\n';
    });
}

HostApiLayer::DriverStatus HostApiLayer::ensure_active_context() const {
    if (!currentContext_) {
        return DriverStatus::error(DriverStatusCode::InvalidContext, "No active context set");
    }
    return DriverStatus::success();
}

}  // namespace cudaway::host
