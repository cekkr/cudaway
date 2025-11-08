# CUDA Validation Programs

These standalone CUDA C++ programs exercise progressively more complex device
features so CUDAway contributors can validate future driver/runtime shims
against real CUDA behavior. Each sample is intentionally tiny, uses
`cuda_runtime.h`, and reports a `[PASS]/[FAIL]` status that higher level
harnesses can scrape.

## Directory Layout

| File | What it covers |
| --- | --- |
| `common.hpp` | Shared helpers for banner printing and `CUDA_CHECK` macro. |
| `01_device_query.cu` | Host-only probe of device inventory and properties. |
| `02_vector_add.cu` | Basic global memory traffic + kernel launch validation. |
| `03_shared_memory_reduction.cu` | Cooperative reduction using shared memory and synchronized warps. |
| `04_matrix_mul_tiled.cu` | Block-tiled GEMM exercising shared memory tiling and boundary guards. |

Add new tests by following the numbering and documenting the covered CUDA
feature in this table so downstream automation can pick them up.

## Building & Running

Any system with the CUDA toolkit installed can build the samples directly with
`nvcc`:

```bash
cd cuda_tests
nvcc 02_vector_add.cu -o vector_add
./vector_add
```

For quick iteration you can compile all tests at once:

```bash
cd cuda_tests
for src in 0*.cu; do
  exe="${src%.cu}"
  nvcc "$src" -o "$exe"
done
```

Integrate the resulting binaries with CUDAway's validation story by launching
them under the LD_PRELOAD/DLL proxy builds once those shims are ready—each test
will fail fast if device interactions deviate from CUDA semantics.
