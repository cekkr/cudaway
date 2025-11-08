# NEXT_STEPS

Central tracker for every "what's next" item scattered through `README.md`, `AI_REFERENCE.md`, and the studies folder. Tasks are ordered by priority/time horizon and cite their original sources so future edits can keep things in sync.

## P0 – Preconditions

1. Confirm the legal go/no-go decision on PTX translation before investing beyond research (studies/FOUNDATIONS.md:59-61).

## P1 – Core Linux Bring-Up (Immediate)

1. Flesh out `HostApiLayer` into a production-grade CUDA Driver/Runtime facade with full symbol coverage, backed by the handle table scaffold (README.md:75-76; studies/FOUNDATIONS.md:66-67).
2. Replace the synthetic PTX compiler with a real PTX → LLVM-IR → AMDGPU backend and wire it into `device::PtxCompiler`, including PTX parsing and binary caching (README.md:76; studies/FOUNDATIONS.md:67; studies/BASE_CONCEPT.md:309).
3. Extend the `studies/` folder with concrete mapping tables (cuBLAS→rocBLAS, cuDNN→MIOpen, cuFFT→rocFFT, etc.) and document ROCm blockers, especially for Windows/private builds (README.md:77-78; studies/FOUNDATIONS.md:68-69).
4. Capture detailed LD_PRELOAD packaging requirements (symbol export lists, build flags, delivery ops note) to unblock the Linux shim rollout (studies/FOUNDATIONS.md:69-70).

## P2 – Tooling & Research Support (Short Term)

1. Draft `tools/specs/ROCmWindowsMatrix.md` so Linux-vs-Windows export gaps and Vulkan fallbacks are formally modeled before coding (AI_REFERENCE.md:97-99).
2. Document parser dependencies (libclang, pycparser fallback) in `tools/README.md` to unblock the header-analysis helpers (AI_REFERENCE.md:99-100).
3. Once the specs are reviewed, scaffold `tools/{specs,python,data}` and introduce a `pyproject.toml` if the helpers share enough utilities to justify it (AI_REFERENCE.md:100-101).

## P3 – Program Roadmap (Sequenced Delivery)

1. **Phase 1 – Linux core:** Ship the LD_PRELOAD layer with Driver API coverage plus the PTX JIT to run canonical Driver API samples (README.md:17-19; studies/BASE_CONCEPT.md:309).
2. **Phase 2 – Runtime & math libraries:** Layer in the Runtime API shim and the cuBLAS/cuSPARSE/cuFFT adapters (README.md:19-20; studies/BASE_CONCEPT.md:310).
3. **Phase 3 – AI/ML focus:** Implement the cuDNN → MIOpen bridge for deep-learning workloads (README.md:20; studies/BASE_CONCEPT.md:311).
4. **Phase 4 – Windows enablement:** Port the stack via DLL proxying and build/distribute a private ROCm stack to compensate for HIP SDK gaps (README.md:21; studies/BASE_CONCEPT.md:312; studies/FOUNDATIONS.md:51-55).
5. **Phase 5 – Scale-out & observability:** Add RCCL/NCCL and RSMI/NVML shims plus hardened caching/tooling for multi-GPU scenarios (README.md:22; studies/BASE_CONCEPT.md:313).

## P4 – Windows Operational Guardrails (Continuous)

1. Keep Linux as the source of truth; run all functional/perf testing on the full ROCm stack where every dependency exists (studies/ROCm-API-LinuxVsWindows.md:326-327).
2. Codify the Windows HIP workflow via the bundled toolchain + `cudaway_hip_windows` helper so contributors all share the same setup (studies/ROCm-API-LinuxVsWindows.md:328-329).
3. Surface runtime diagnostics that probe for `amdhip64.dll`, `hiprtc.dll`, and math libraries at startup, guiding users to set `CUDAWAY_ROCM_WINDOWS_ROOT` (studies/ROCm-API-LinuxVsWindows.md:330-332).
4. Document when to fall back to WSL2 so contributors know when the native Windows path is insufficient (studies/ROCm-API-LinuxVsWindows.md:333-334).
