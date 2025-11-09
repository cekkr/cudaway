# AI Reference

Fast-access knowledge base for CUDAway. Update this alongside `README.md` whenever code or docs
change so the next agent inherits the latest context.

## Collaboration Rules

- Mirror meaningful changes into `README.md`, `AI_REFERENCE.md`, and (if a study) `studies/*`.
- Record research results under `studies/` and cross-link them here so we avoid re-running the same
  investigations.
- Work incrementally, document breaking changes, and run the available build/test commands before
  yielding control.

## Latest Context (2025-02-16)

- Host layer is now a typed CUDA Driver facade: `HostApiLayer` exposes `DriverStatus` return values,
  tracks primary/active contexts, and records per-module compilation metadata (cache key/path) in
  the handle tables. CLI smoke tests fail fast if any step in the init→module→function→launch chain
  reports a non-success code.
- `src/host/CudaDriverShim.*` exports the initial CUDA Driver API surface (`cuInit`, `cuDevice*`,
  `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`) and routes each entry point into
  `HostApiLayer`. The CLI now drives the same path that the upcoming LD_PRELOAD/DLL proxy builds
  will expose, so future work can focus on HIP wiring instead of ad-hoc harnesses.
- `device::PtxCompiler` computes a deterministic FNV-1a digest for each PTX payload, serves
  in-memory + on-disk caches (under `$TMPDIR/cudaway_ptx_cache` by default), and hands cache hits
  back to the host layer so future LLVM integration can piggy-back on the same metadata.
- `NEXT_STEPS.md` was expanded with deliverable/validation bullets per priority item so reviewers
  can tell when the Driver API hardening and PTX compiler work are "done enough" for the next phase.
- Bootstrapped a CMake-based C++20 skeleton:
  - `src/host/HostApiLayer.*` – Driver API facade that currently logs lifecycle events and manages
    contexts/module/function handles.
  - `src/device/PtxCompiler.*` – PTX→GCN compiler scaffold, still synthetic but cache-aware.
  - `src/common/HandleTable.hpp` & `Types.hpp` – shared utilities for opaque handle management.
  - `src/main.cpp` – CLI stub wiring the layers together; useful for smoke tests.
- Added `studies/FOUNDATIONS.md` with a concise derivation of the Project Cerberus blueprint.
- `cmake` configure + build verified locally (`cmake -S . -B build && cmake --build build`).
- Added platform detection + HIP runtime discovery (`src/platform/PlatformConfig.*`) so the host
  layer logs actionable hints on Linux vs. Windows environments.
- Introduced `cmake/HipWindowsWorkarounds.cmake` and the
  `cmake/toolchains/windows-hip.cmake` helper to codify the Windows HIP SDK workaround; on Windows
  configure with `-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-hip.cmake`.
- Added `cuda_tests/` with four increasing-complexity CUDA runtime samples (device query, vector add,
  shared-memory reduction, tiled GEMM) so the compatibility layer can be validated against native
  NVIDIA behavior.
- Bootstrapped `tools/python/cuda_runtime_converter.py` plus the supporting `tools/data/` cache so
  CUDA Runtime↔HIP mapping data and `src/host/runtime/RuntimeStubTable.generated.hpp` can be
  regenerated directly from the canonical PDFs. The converter now tags entries as `hip-documented`
  vs. `needs-shim` and emits a status breakdown in the JSON metadata.
- Built `tools/python/cuda_rocm_api_generator.py` + `tools/data/cuda_rocm_driver_apis.json`; the
  script emits `src/host/generated/CudaRocmApi.generated.hpp` so each CUDA Driver API entry point,
  its ROCm analogue, and the associated input/output conversion placeholders stay in sync (it even
  pre-stubs future cases like BF16 emulation). Re-run the generator whenever you touch the driver
  shim surface.
- `src/host/runtime/RuntimeRegistry.*` consumes the generated table, exposes lookup helpers, and
  prints the HIP-documented vs. needs-shim counts during CLI runs so regressions in the mapping data
  are highly visible.
- `tools/README.md` documents the parser dependencies (pypdf, libclang, pycparser) so header-
  analysis helpers have a clear onboarding story.
- Seeded library coverage tables in `studies/mappings/CUDA_ROCM_Library_Mappings.md` (cuBLAS,
  cuDNN, cuFFT) with status flags and Windows blockers, and linked them from README/NEXT_STEPS to
  guide upcoming mapping automation.
- Captured the Linux LD_PRELOAD packaging blueprint in `studies/LD_PRELOAD_PACKAGING.md`, detailing
  symbol exports, linker flags, release layout, and validation so CMake packaging can follow suit.

## Build & Test Checklist

```bash
cmake -S . -B build       # configure project
cmake --build build       # compile cudaway + libcudaway_core
./build/cudaway           # smoke test host/device scaffolding
```
Windows HIP SDK path: `cmake -S . -B build-windows -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-hip.cmake -DCUDAWAY_ROCM_WINDOWS_ROOT="C:/Program Files/AMD/ROCm"`.

All new code should compile cleanly with `-Wall -Wextra -Wpedantic` (set in CMake). Add targeted
unit/system tests as we introduce real functionality.

## Core Code Map

- `CMakeLists.txt` – defines `cudaway_core` static library and `cudaway` CLI, enforces C++20.
- `src/common/HandleTable.hpp` – threadsafe map generating opaque CUDA-style handles.
- `src/host/HostApiLayer.*` – shim entry point that now exposes typed `DriverStatus` values,
  primary/active context registries, and handle-table backed module/function objects ready for HIP
  wiring.
- `src/device/PtxCompiler.*` – PTX compiler scaffold featuring deterministic cache keys plus
  memory/disk caches; swap the synthetic binary emitter with the PTX→LLVM→AMDGPU pipeline next.
- `src/platform/PlatformConfig.*` – detects host OS, discovers HIP installation roots, and prints the
  correct preload vs. DLL proxy workflow plus search paths at runtime.
- `studies/BASE_CONCEPT.md` – original Project Cerberus research (very long, mostly single-line).
- `studies/FOUNDATIONS.md` – working summary + actionable tasks derived from the base concept.

## Gaps Worth Watching

- Host API still omits real HIP device/context/module calls; only the bookkeeping/logging path is in
  place, so next steps are wiring HIP entry points and exporting actual CUDA symbols.
- PTX compiler continues to emit synthetic binaries despite the new cache plumbing; it still needs a
  real PTX parser, LLVM AMDGPU backend, and disk cache eviction strategy.
- Ecosystem shims (cuBLAS→rocBLAS, etc.) have not been started; see the mappings outlined in
  `studies/FOUNDATIONS.md`.
- Windows support requires packaging private ROCm builds (large engineering lift).

## Tooling Automation Evaluation (2025-02-15)

We want a `tools/` workspace that allows Markdown specs to land before Python helpers. The aim is to remove repetitive research/build chores, surface ROCm feature gaps promptly, and keep future agents fed with structured data.

### Goals

- Capture design notes under `tools/specs/*.md` ahead of code so reviewers can validate scope, dependencies, and tests.
- Keep production helpers under `tools/python/` with lightweight CLI wrappers (`python -m tools.python.<tool>`).
- Mirror decisive findings into `AI_REFERENCE.md` / `studies/*.md` to preserve the research trail.

### Proposed Layout

- `tools/README.md` – onboarding doc covering dependencies (Python 3.11+, `libclang`, optional Vulkan SDK) and usage.
- `tools/specs/*.md` – pre-implementation briefs such as `ROCmWindowsMatrix.md`, `CudaRocmMapper.md`, `BinderEvaluator.md`.
- `tools/python/` – namespace packages for the actual scripts plus shared utilities under `tools/python/common`.
- `tools/data/` – cached JSON/YAML describing harvested APIs so expensive header scans remain optional.

### Candidate Tools

1. `tools/python/rocm_windows_matrix.py`
   - Purpose: Compare ROCm Linux vs. Windows exports (headers, DLLs) and emit a coverage report/JSON manifest. Highlights functions needing HIP SDK packaging vs. Vulkan-backed fallback paths when Windows ROCm is too limited.
   - Inputs: ROCm install roots, optional Vulkan SDK path for capability probing on Windows.
   - Outputs: Markdown summary plus a machine-readable manifest feeding `studies/ROCm-API-LinuxVsWindows.md`.
2. `tools/python/cuda_rocm_mapper.py`
   - Purpose: Parse CUDA driver/runtime headers through `clang.cindex`, match symbols against ROCm or shim implementations, and generate mapping tables plus skeleton binders for `src/host/HostApiLayer`.
   - Outputs: `tools/data/cuda_to_rocm.json` and optional header templates for cuBLAS/cuDNN adapters.
3. `tools/python/binder_evaluator.py`
   - Purpose: Consume the mapping data, diff it against the actual host-layer coverage, and produce progress reports (percent implemented, missing enums, etc.).
4. `tools/python/build_doctor.py`
   - Purpose: Validate HIP/LLVM/Vulkan toolchains across Linux/Windows, surface remediation steps, and optionally patch the `cmake/toolchains` cache entries for contributors.

Each helper starts as a spec (`tools/specs/ToolName.md`) detailing inputs, outputs, data sources, and validation. After sign-off we add the Python module, minimal unit tests, and link it from `tools/README.md`.

**Update (Runtime focus):** `tools/python/cuda_runtime_converter.py` already ingests the CUDA Runtime API + HIP programming guide PDFs, emits `tools/data/cuda_runtime_mappings.json`, and auto-generates the host-facing stub table at `src/host/runtime/RuntimeStubTable.generated.hpp`.

### Next Steps

`NEXT_STEPS.md` is now the single backlog source; consult it for active tooling tasks and update that file first before mirroring high-level context here. Remove already completed and no more important steps to remember.


## Research Archive (studies/)

- `BASE_CONCEPT.md` – authoritative technical blueprint + risk analysis.
- `FOUNDATIONS.md` – condensed notes, TODOs, and cross references to source scaffolding.
- `ROCm-API-LinuxVsWindows.md` – deep dive into feature/library gaps, Windows workarounds, and the
  scripted CMake toolchain approach now codified under `cmake/`.
- `mappings/CUDA_ROCM_Library_Mappings.md` – living table of cuBLAS/cuDNN/cuFFT symbol status and
  action items feeding `tools/data/` plus Windows availability flags.
- `LD_PRELOAD_PACKAGING.md` – operational checklist for building and distributing the Linux shim
  (symbols, linker flags, artifact layout, release validation).

Keep this index synchronized when new material lands in `studies/`.

## When Starting a New Task

- Skim TODOs inside `src/host` and `src/device`.
- Check whether the study notes already cover your topic; extend them if not.
- Define how you will test the change (unit tests, integration harness, or CLI smoke test).

## Contact & Credits

- Project owner: Riccardo Cecchini (Mozilla Public License Version 2.0, 2025).

If any workflow rule changes, reflect it here immediately so human collaborators and AI agents stay
aligned.
