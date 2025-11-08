# CUDA ↔ ROCm Library Mappings

This note seeds the mapping tables requested in `NEXT_STEPS.md` so host shims know which CUDA
symbols should forward to an existing ROCm implementation versus require bespoke fallbacks. Each
section highlights the highest-traffic entry points, identifies ROCm coverage or gaps (especially on
Windows per `studies/ROCm-API-LinuxVsWindows.md`), and cites the data sources that future automation
should ingest into `tools/data/`.

Status legend:

| Status  | Meaning |
|---------|---------|
| ✅ Ready | Entry point exists in ROCm and matches CUDA semantics with minor handle translation. |
| ⚠️ Shim  | Requires thin shim (e.g., enum remap, descriptor translation) but backend exists. |
| ⛔ Blocked | No ROCm equivalent; requires fallback or is missing in public Windows builds. |

## cuBLAS ↔ rocBLAS

Source: CUDA 12.3 cuBLAS manual + ROCm 6.1 rocBLAS documentation.

| Category | CUDA Symbol | ROCm Symbol | Status | Notes |
|----------|-------------|-------------|--------|-------|
| Handle   | `cublasCreate_v2` | `rocblas_create_handle` | ✅ | Translate return codes to `DriverStatus`; handles live inside `common::HandleTable`. |
| Handle   | `cublasDestroy_v2` | `rocblas_destroy_handle` | ✅ | Needs safe teardown tied to HIP context lifetime. |
| Stream   | `cublasSetStream_v2` | `rocblas_set_stream` | ⚠️ | rocBLAS enforces hipStream_t; need host-layer stream registry. |
| GEMM     | `cublasSgemm` | `rocblas_sgemm` | ✅ | Parameter order matches; alpha/beta pointers remain device pointers. |
| GEMM     | `cublasGemmEx` | `rocblas_gemm_ex` | ⚠️ | rocBLAS lacks tensor op fallback on MI200 pre-RDNA3; requires datatype negotiation. |
| Batched  | `cublasSgemmBatched` | `rocblas_sgemm_batched` | ⚠️ | rocBLAS expects array-of-pointers; CUDA apps may rely on strided variant. |
| Lt API   | `cublasLtMatmul` | (none) | ⛔ | rocBLAS Lt equivalent is still experimental, absent from Windows SDK. |

Action items:

- Encode the above rows plus the remaining GEMV/AXPY/solver ops into `tools/data/cuBLAS_mappings.csv`.
- Extend `studies/ROCm-API-LinuxVsWindows.md` with Lt status once AMD publishes Windows coverage.
- Attach per-symbol validation steps (sample size, alignment constraints) before wiring host shims.

## cuDNN ↔ MIOpen

Source: CUDA 9–12 cuDNN developer guide + ROCm 6.1 MIOpen reference.

| Category | CUDA Symbol | ROCm Symbol | Status | Notes |
|----------|-------------|-------------|--------|-------|
| Handle   | `cudnnCreate` | `miopenCreate` | ✅ | Handles scoped by HIP context; prefer RAII wrapper. |
| Tensor   | `cudnnCreateTensorDescriptor` | `miopenCreateTensorDescriptor` | ⚠️ | Rank/layout enums differ; need conversion utilities. |
| Conv     | `cudnnConvolutionForward` | `miopenConvolutionForward` | ⚠️ | Algorithm selection APIs diverge; rely on `miopenFindConvolutionForwardAlgorithm`. |
| Conv     | `cudnnGetConvolutionForwardWorkspaceSize` | `miopenConvolutionForwardGetWorkSpaceSize` | ✅ | Size units align; still return via `size_t`. |
| RNN      | `cudnnRNNForwardTraining` | (none) | ⛔ | MIOpen lacks feature-complete RNN API; must fall back to host-emulation or hipCUBLASLt. |
| Norm     | `cudnnBatchNormalizationForwardInference` | `miopenBatchNormalizationForwardInference` | ✅ | Parameter ordering identical; only enum remap required. |

Action items:

- Enumerate all descriptor types (tensor, filter, convolution, activation) and capture enum/value
  deltas in `tools/data/cudnn_descriptor_mappings.json`.
- Document Windows availability: public HIP SDK omits MIOpen, so all entries are effectively ⛔ on
  Windows until a private ROCm build ships (see `studies/ROCm-API-LinuxVsWindows.md`).

## cuFFT ↔ rocFFT

Source: CUDA 12 cuFFT API guide + ROCm 6.1 rocFFT API reference.

| Category | CUDA Symbol | ROCm Symbol | Status | Notes |
|----------|-------------|-------------|--------|-------|
| Plan     | `cufftPlan1d` | `rocfft_plan_create` | ⚠️ | rocFFT splits plan + execution descriptors; wrap in helper struct. |
| Plan     | `cufftDestroy` | `rocfft_plan_destroy` | ✅ | Tie destruction to host handle table to avoid double-free. |
| Exec     | `cufftExecC2C` | `rocfft_execute` | ⚠️ | rocFFT requires explicit execution info object; auto-create per launch. |
| Stream   | `cufftSetStream` | `rocfft_execution_info_set_stream` | ⚠️ | Execution info must be cached per plan; add LRU to avoid churn. |
| Xt API   | `cufftXtSetCallback` | (none) | ⛔ | Callback hooks absent; warn users and no-op with diagnostics. |

Action items:

- Prototype CSV schema (`tools/data/cufft_plan_mappings.csv`) capturing dimensionality, precision,
  and stride support so the host shim can validate unsupported combos early.
- Flag Windows gaps: rocFFT DLLs are missing from the public HIP SDK; see Section 4.2 of
  `studies/ROCm-API-LinuxVsWindows.md`.

## Shared Next Steps

1. Feed these seed tables into `tools/specs/CudaRocmMapper.md` once that spec lands; Python helpers
   can then regenerate machine-readable manifests.
2. Expand coverage to cuSPARSE and NCCL/RCCL using the same status legend so the backlog has
   consistent signals.
3. Reference this note from `README.md` + `AI_REFERENCE.md` to keep collaborators on the same mapping
   truth set.
