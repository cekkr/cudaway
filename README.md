# CUDAway

Experimental CUDA-to-ROCm translation layer inspired by the **Project Cerberus** study
(`studies/BASE_CONCEPT.md`). The goal is to let unmodified CUDA applications run on AMD GPUs by
intercepting host API calls and JIT-compiling PTX kernels to AMD GCN/RDNA binaries.

## Architectural Snapshot

- **Host API Layer (src/host)** – provides libcuda/libcudart-compatible entry points, maps CUDA
  handles to HIP objects, and orchestrates kernel/module lifecycles.
- **Device Kernel Layer (src/device)** – will embed a PTX → LLVM-IR → AMDGPU JIT compiler.
- **Handle & state utilities (src/common)** – shared infrastructure for maintaining opaque handles
  that mimic CUDA driver/runtime behaviour.
- **Ecosystem shims** – future components will forward cuBLAS/cuDNN/cuFFT/cuSPARSE/NCCL/NVML calls
  to their ROCm counterparts, guided by the mappings catalogued in `studies/BASE_CONCEPT.md`.

High-level roadmap (see Section 7.2 of the study):
1. Ship the Linux LD_PRELOAD layer with Driver API coverage and the PTX JIT.
2. Layer in the Runtime API and math/FFT shims.
3. Implement cuDNN → MIOpen for AI workloads.
4. Port the stack to Windows via DLL proxying and a private ROCm distribution.
5. Add scale-out tooling (RCCL, RSMI) and hardened caching.

## Repository Layout

```
.
├── CMakeLists.txt        # Top-level configuration (C++20, static core library + CLI stub)
├── src
│   ├── common            # Handle tables and strongly typed handles
│   ├── device            # PTX compiler scaffold
│   ├── host              # Host API translation scaffold
│   └── main.cpp          # Simple driver demonstrating flow
└── studies               # Research artefacts (BASE_CONCEPT + derived notes)
```

See `studies/FOUNDATIONS.md` for a concise, working-summary of the translation plan extracted from
the detailed base concept.

## Building & Running

```bash
cmake -S . -B build
cmake --build build
./build/cudaway
```

The stub binary initialises the host layer, compiles a sample PTX payload, and launches a mock
kernel—useful for validating the build system and dependency graph before wiring in HIP/LLVM.

## Next Steps

- Flesh out the Host API shim so it exposes the actual CUDA driver/runtime symbol surface.
- Embed a real PTX parser + LLVM AMDGPU backend inside `device::PtxCompiler`.
- Extend the studies folder with concrete mapping tables (cuBLAS→rocBLAS, cuDNN→MIOpen, etc.) and
  record blockers for Windows where a private ROCm build is required.
