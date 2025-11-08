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

## Project Status (Feb 2025)

- `src/host` exposes a skeletal CUDA Driver API facade that manages opaque module/function handles.
- `src/device` contains a PTX compiler scaffold ready to host the LLVM AMDGPU backend.
- `src/platform` detects HIP installations, validates DLL/SO presence, and prints preload/proxy
  instructions for each OS.
- CMake builds a static `cudaway_core` library plus a CLI stub (`src/main.cpp`) used for smoke tests.

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
contributors can quickly see which CUDA feature maps where in ROCm.

## Platform & Tooling Story

**Linux**
- Primary development target; LD_PRELOAD layer will intercept CUDA calls and route them through HIP.
- Uses system ROCm packages; `src/platform/PlatformConfig` verifies required `.so` files before
  enabling the shim.

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
