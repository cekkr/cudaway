# CUDAway Tooling

Helper scripts that keep the research data, host shims, and generated bindings in sync. Python 3.9+
with `pypdf` is required. Install deps via:

```bash
python3 -m pip install --user -r <(printf "pypdf\n")
```

> Prefer a proper environment manager? `pyproject.toml` at repo root declares the same dependency.

## cuda_runtime_converter

Parses `studies/APIs/CUDA_Runtime_API.pdf` and `studies/APIs/AMD_HIP_Programming_Guide.pdf`,
builds CUDA↔HIP mapping data, and emits both JSON (`tools/data/cuda_runtime_mappings.json`) and a
C++ stub table (`src/host/runtime/RuntimeStubTable.generated.hpp`).

```bash
python3 -m tools.python.cuda_runtime_converter \
  --cuda-pdf studies/APIs/CUDA_Runtime_API.pdf \
  --hip-pdf studies/APIs/AMD_HIP_Programming_Guide.pdf
```

Use `--default-status` to annotate every entry (e.g. `implemented`, `pending-hip`, etc.). Re-run the
converter whenever the PDFs change so host/runtime adapters never drift from the published APIs.
