# CUDAway

> CUDAway is an experimental CUDA-to-ROCm compatibility layer inspired by the **Project Cerberus**
> research (`studies/BASE_CONCEPT.md`). It aims to run unmodified CUDA applications on AMD GPUs by
> intercepting host API calls and JIT-compiling PTX kernels into AMD GCN/RDNA binaries.

## Why CUDAway?

- **Drop-in host shims** – mimic `libcuda`/`libcudart` entry points so existing binaries can be
  preloaded (Linux) or proxied (Windows) without relinking.
- **Real PTX execution path** – planned PTX → LLVM IR → AMDGPU codegen pipe lets CUDA kernels launch
  on HIP runtimes instead of via software emulation.
- **Cross-platform story** – Linux-first LD_PRELOAD flow plus a Windows toolchain (`cmake/toolchains`)
  and runtime discovery logic (`src/platform`) keep both ecosystems in lockstep.
- **Research-grounded** – every architectural trade-off is documented under `studies/`, making it
  easy to trace design choices back to Cerberus findings.

## Current Status (Feb 2025)

- `HostApiLayer` is now a typed Driver API facade that returns explicit `DriverStatus` values,
  keeps primary vs. active contexts straight, and records module/function metadata so cache hits in
  later PTX compilations can be traced. CLI smoke tests walk the init → module → function → launch
  chain and fail fast when any driver call reports a non-success status.
- `src/host/CudaDriverShim.*` now exports a starter set of CUDA Driver API symbols
  (`cuInit`/`cuDevice*`/`cuModuleLoadData`/`cuLaunchKernel`) so the future LD_PRELOAD/DLL proxy
  layers call into the existing `HostApiLayer` bookkeeping. The CLI binary already exercises these
  exports end-to-end, proving the wrapper wiring works ahead of the HIP bindings.
- `device::PtxCompiler` computes an FNV-1a digest per PTX payload, serves both in-memory and
  `$TMPDIR/cudaway_ptx_cache` hits, and persists compilation metadata needed for the upcoming LLVM
  AMDGPU backend. It still emits synthetic binaries, but running the CLI twice already demonstrates
  cache reuse vs. recompile penalties.
- `src/platform/PlatformConfig.*` now auto-detects OS + HIP installs, probes for the expected
  `.so`/`.dll` inventory, and prints actionable preload/proxy guidance so Linux and Windows setups
  share the same troubleshooting flow.
- The Windows toolchain files (`cmake/HipWindowsWorkarounds.cmake`,
  `cmake/toolchains/windows-hip.cmake`) codify the `cudaway_hip_windows` helper target and standard
  HIP SDK search paths. Configure via `-DCUDAWAY_ROCM_WINDOWS_ROOT="C:/Program Files/AMD/ROCm"` to
  keep contributors on identical flags.
- Tooling bootstrap: `tools/python/cuda_runtime_converter.py` ingests the CUDA Runtime API and HIP
  Programming Guide PDFs, regenerates `tools/data/cuda_runtime_mappings.json`, and emits
  `src/host/runtime/RuntimeStubTable.generated.hpp` so runtime symbols never drift from the spec.
- Runtime stub coverage is now surfaced directly in the CLI: `RuntimeRegistry` consumes the
  generated table and prints how many entries are HIP-documented versus still needing shims,
  providing instant feedback when the mapping data changes.
- The top-level CMake configure (`cmake -S . -B build`) and build (`cmake --build build`) flow is
  green on Linux; use `./build/cudaway` for regression smoke tests. Windows follows the Ninja
  workflow outlined below once the HIP SDK is installed.

## Next Steps Snapshot

`NEXT_STEPS.md` tracks the live backlog; highlights for quick human onboarding:

- **P0 – Legal checkpoint:** resolve the PTX translation go/no-go so implementation work scales with
  confidence.
- **P1 – Core bring-up:** finish wiring the Driver API stubs (init → launch path with typed
  bookkeeping), swap the synthetic PTX emitter for the LLVM-backed pipeline with disk caching, grow
  the studies-backed CUDA↔ROCm mapping tables, and document the Linux LD_PRELOAD packaging recipe.
- **P2 – Tooling support:** land the `tools/specs/ROCmWindowsMatrix.md` brief, call out parser deps
  in `tools/README.md`, and scaffold the shared `tools/{specs,python,data}` workspace now that the
  runtime converter is live.
- **P3–P5 – Roadmap guardrails:** execute the staged Linux→Runtime→AI→Windows→Scale rollout, keeping
  Windows guardrails (HIP diagnostics, WSL2 fallback criteria, shared toolchain usage) documented as
  detailed in `NEXT_STEPS.md` and `studies/ROCm-API-LinuxVsWindows.md`.

## Architecture at a Glance

- **Host API Layer (`src/host`)** – translates CUDA Driver/Runtime calls into HIP operations, owns
  handle lifecycles, and will multiplex ecosystem shims (cuBLAS, cuDNN, etc.).
- **Device Kernel Layer (`src/device`)** – future PTX parser + LLVM AMDGPU backend that emits native
  binaries and caches them per-module.
- **Common utilities (`src/common`)** – strongly typed handle tables shared across shims.
- **Ecosystem shims** – planned adapters that forward CUDA libraries to their ROCm counterparts using
  the mapping tables maintained in `studies/`.

## CUDA ↔ ROCm Cheat Sheet

| CUDA surface                | ROCm / AMD counterpart        | Status / Notes |
| --------------------------- | ----------------------------- | -------------- |
| CUDA Driver API (`cu*`)     | HIP Driver API (`hip*`)       | Host shim scaffolded; needs full symbol export + error parity. |
| CUDA Runtime API (`cuda*`)  | HIP Runtime API               | Planned after Driver API hardening. |
| cuBLAS                      | rocBLAS                       | Mapping table in progress (`studies/FOUNDATIONS.md`). |
| cuDNN                       | MIOpen                        | Required for AI workloads; depends on tensor layout adapters. |
| cuFFT                       | rocFFT                        | To be layered once PTX compiler handles math intrinsics. |
| cuSPARSE                    | rocSPARSE                     | Similar shim pattern to cuBLAS. |
| NCCL                        | RCCL                          | Needed for scale-out; Windows availability tracked in studies. |
| NVML / CUPTI instrumentation| RSMI / rocProfiler            | Research ongoing; informs telemetry strategy. |

The table above mirrors the living research doc (`studies/ROCm-API-LinuxVsWindows.md`) so
contributors can quickly see which CUDA feature maps where in ROCm. For per-symbol coverage, status
flags, and Windows availability, consult `studies/mappings/CUDA_ROCM_Library_Mappings.md`.

## Platform & Tooling Story

**Linux**
- Primary development target; LD_PRELOAD layer will intercept CUDA calls and route them through HIP.
- Uses system ROCm packages; `src/platform/PlatformConfig` verifies required `.so` files before
  enabling the shim.
- Packaging checklist (symbols, linker flags, release layout) now lives in
  `studies/LD_PRELOAD_PACKAGING.md`.

**Windows**
- Ships `cmake/toolchains/windows-hip.cmake` plus `cmake/HipWindowsWorkarounds.cmake` to emulate the
  Linux `hip::host` targets while HIP's official CMake integration matures.
- Set `CUDAWAY_ROCM_WINDOWS_ROOT` (defaults to `C:/Program Files/AMD/ROCm`) so the build locates
  headers, `amdhip64.dll`, and companion libraries.
- Runtime proxying will rely on DLL forwarders instead of LD_PRELOAD; the platform layer validates
  that the expected HIP DLLs exist and prints remediation hints.

## Repository Layout

```
.
├── CMakeLists.txt        # C++20 build: libcudaway_core + cudaway CLI
├── src
│   ├── common            # Handle tables / strongly typed IDs
│   ├── device            # PTX compiler scaffold
│   ├── host              # CUDA Driver shim entry points
│   ├── platform          # OS + HIP discovery helpers
│   └── main.cpp          # CLI smoke test wiring the layers
├── tools                 # Python helpers + generated data/stub outputs
├── cmake                 # HIP workarounds + Windows toolchains
└── studies               # Project Cerberus research + derived notes
```

See `studies/FOUNDATIONS.md` for the concise summary of the broader blueprint, and
`AI_REFERENCE.md` for fast-moving implementation notes.

## Build & Run

```bash
cmake -S . -B build
cmake --build build
./build/cudaway
```

The CLI stub initialises the host layer, compiles a sample PTX payload, and launches a mock kernel—
use it as a quick integration test each time you touch the build graph.

### CUDA Validation Samples

The new `cuda_tests/` directory contains a numbered suite of standalone CUDA
programs (device query, vector add, shared-memory reduction, tiled GEMM) that
progress from basic runtime usage to more complex shared-memory patterns. Build
them with `nvcc 0X_*.cu -o <name>` on a CUDA-enabled system, then run the
resulting binaries directly or under the CUDAway LD_PRELOAD/DLL proxy to
validate host-driver behavior against NVIDIA's runtime.

### Windows HIP SDK Workflow

Native HIP CMake support on Windows is still catching up, so CUDAway ships a scripted workaround:

1. Install the AMD HIP SDK and (optionally) set `CUDAWAY_ROCM_WINDOWS_ROOT` to the install path.
2. Configure with the bundled toolchain:
   ```bash
   cmake -S . -B build-windows -G Ninja \
     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-hip.cmake \
     -DCUDAWAY_ROCM_WINDOWS_ROOT="C:/Program Files/AMD/ROCm"
   ```
3. Build via `cmake --build build-windows`.

The toolchain wires `${HIP_ROOT}/include` and `lib`, links against `amdhip64`, and defines the
`cudaway_hip_windows` interface target so contributors share one set of HIP flags.

## Roadmap Highlights

1. Ship the Linux LD_PRELOAD layer with Driver API coverage and the PTX JIT.
2. Layer in the Runtime API plus math/FFT shims (cuBLAS/rocBLAS, cuFFT/rocFFT, etc.).
3. Implement cuDNN → MIOpen adapters for AI workloads.
4. Port the stack to Windows via DLL proxying and a private ROCm distribution.
5. Add scale-out tooling (RCCL, RSMI) and hardened code-object caching.

Track day-to-day priorities in `NEXT_STEPS.md`; update `AI_REFERENCE.md` whenever you land notable
code or research so future contributors inherit the latest context.

### Tooling & Automation

- `tools/python/cuda_runtime_converter.py` ingests the CUDA Runtime + HIP programming guide PDFs and
  generates `tools/data/cuda_runtime_mappings.json` plus the host-facing
  `src/host/runtime/RuntimeStubTable.generated.hpp`. Re-run it whenever the upstream docs rev so the
  HostApiLayer/Runtime surface never drifts from official APIs. The converter now tags entries as
  `hip-documented` versus `needs-shim`, and the CLI summarizes the resulting status breakdown at run
  time via `RuntimeRegistry`.
