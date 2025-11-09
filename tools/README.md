# CUDAway Tooling

Helper scripts that keep the research data, host shims, and generated bindings in sync.

## Dependencies

- Python 3.9+.
- [`pypdf`](https://pypi.org/project/pypdf/) for PDF parsing (install via `pip install pypdf` or
  `pyproject.toml` at repo root).
- `libclang` + Python bindings (LLVM 16+) for the upcoming header-analysis helpers such as
  `cuda_rocm_mapper`; set `LLVM_CONFIG`/`CLANG_LIBRARY_PATH` so `clang.cindex` can locate the shared
  library.
- [`pycparser`](https://pypi.org/project/pycparser/) is the fallback parser when libclang is absent;
  it enables basic symbol extraction at the cost of C++ macro hygiene.

> Prefer a proper environment manager? `pyproject.toml` at repo root declares the same dependency
> set. For macOS/Linux, install `libclang` via your package manager (`apt install clang`,
> `brew install llvm`). Windows contributors should install the LLVM binaries and expose
> `LLVM_INSTALL_DIR`.

```bash
python3 -m pip install --user pypdf pycparser
```

## cuda_runtime_converter

Parses `studies/APIs/CUDA_Runtime_API.pdf` and `studies/APIs/AMD_HIP_Programming_Guide.pdf`,
builds CUDA↔HIP mapping data, and emits both JSON (`tools/data/cuda_runtime_mappings.json`) and a
C++ stub table (`src/host/runtime/RuntimeStubTable.generated.hpp`).

```bash
python3 -m tools.python.cuda_runtime_converter \
  --cuda-pdf studies/APIs/CUDA_Runtime_API.pdf \
  --hip-pdf studies/APIs/AMD_HIP_Programming_Guide.pdf \
  --hip-documented-status hip-doc \
  --hip-missing-status needs-shim
```

Use `--default-status` to override the fallback annotation if neither rule matches. The documented vs.
missing status labels propagate into both the JSON metadata (`status_breakdown`) and the generated
C++ table so host/runtime adapters can surface progress dashboards. Re-run the converter whenever the
PDFs change so host/runtime adapters never drift from the published APIs.

## cuda_rocm_api_generator

Reads `tools/data/cuda_rocm_driver_apis.json` and emits
`src/host/generated/CudaRocmApi.generated.hpp`, which catalogues every CUDA Driver API entry point,
its ROCm equivalent, and the per-parameter conversion placeholders (input/output/return macros
included). The generated header also provides placeholder HIP/CUDA type aliases so the codebase can
compile before native headers are wired in, plus BF16 conversion stubs for future emulation work.

```bash
python3 -m tools.python.cuda_rocm_api_generator \
  --spec tools/data/cuda_rocm_driver_apis.json \
  --header-out src/host/generated/CudaRocmApi.generated.hpp
```

Extend the JSON whenever driver coverage grows and re-run the generator so the metadata table and
conversion macros stay synchronized with `src/host/CudaDriverShim.*` and upcoming ROCm bindings.
