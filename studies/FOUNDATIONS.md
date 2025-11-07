# CUDAway Foundations

This note distills the actionable pieces of `studies/BASE_CONCEPT.md` and maps them to the initial
code structure now living under `src/`.

## Mission Profile

- **Goal** – Transparent CUDA-on-ROCm execution (binary-only apps, no recompilation).
- **Strategy** – Dual-pipeline design:
  1. Host shims export CUDA Driver/Runtime symbols and translate calls to HIP/ROCm equivalents.
  2. Device JIT ingests PTX, emits LLVM-IR, and lowers to AMD GCN/RDNA binaries on demand.

## Host API Translation Requirements

- Prefer Driver API interception to guarantee coverage even for Runtime apps (Runtime ultimately
  falls through to the Driver API).
- Linux delivery vehicle: single `libcr-ttl.so` loaded via `LD_PRELOAD`, exporting the complete set
  of CUDA + CUDA library symbols.
- Windows delivery vehicle: proxy DLLs (nvcuda.dll, cudart*.dll, cublas*.dll, …) placed next to the
  target executable to exploit the DLL search order.
- Maintain 1:1 handle maps for contexts, modules, functions, streams, events. The new
  `common::HandleTable` scaffolding implements the required infrastructure.

## Device Kernel JIT Pipeline

1. **PTX ingestion** – parse PTX into an AST using a redistributable parser (e.g. `ptx-parser`).
2. **Semantic lowering** – walk the AST, emit SSA-friendly LLVM-IR, and respect CUDA address spaces
   (global/shared/local).
3. **Codegen** – run LLVM AMDGPU passes, emit binary, and cache by PTX hash before calling
   `hipModuleLoadData`.

`device::PtxCompiler` presently returns a synthetic binary so the host layer can be exercised; this
is the attachment point for the full LLVM toolchain.

## Ecosystem Library Coverage

| CUDA Library | ROCm Target | Notes |
|--------------|-------------|-------|
| cuBLAS       | rocBLAS     | Straightforward handle translation (see HIPIFY tables). |
| cuSPARSE     | rocSPARSE   | Map descriptors directly to avoid hipSPARSE double translation. |
| cuFFT        | rocFFT      | Both are plan-based; reuse handle table utilities. |
| cuDNN        | MIOpen      | Largest surface; descriptors & enums must be mirrored. |
| NCCL         | RCCL        | Binary compatible; mostly pass-through. |
| NVML         | RSMI        | Stateless monitoring shim. |

Each shim should live under `src/host/lib/<name>` once development begins, with dedicated study
notes capturing enum/function deltas.

## Windows Considerations

- Public Windows HIP SDK lacks features needed for this project (no training, partial libraries,
  no distributed support).
- Mitigation: build and ship a private ROCm stack (rocBLAS, MIOpen, rocFFT, rocSPARSE, RCCL, RSMI)
  using `hip-clang` + Visual Studio + Ninja. This is a major engineering effort that should start
  only after the Linux LD_PRELOAD path is validated.

## Legal & Technical Risks

- NVIDIA EULA explicitly forbids translating PTX to run on non-NVIDIA hardware—decide on the
  go/no-go posture before investing beyond research.
- Sustaining a Windows-only ROCm fork represents a high maintenance cost.
- JIT warm-up latency must be mitigated via persistent caching.

## Immediate Work Items

1. Expand `HostApiLayer` into a proper Driver API facade, backed by the handle table scaffold.
2. Replace the synthetic PTX compiler with an LLVM-powered implementation (`device::PtxCompiler`).
3. Document concrete mapping tables for cuBLAS/cuDNN/cuFFT in `studies/` to guide shim development.
4. Capture requirements for LD_PRELOAD packaging (symbol export lists, build flags) in a dedicated
   ops note.
