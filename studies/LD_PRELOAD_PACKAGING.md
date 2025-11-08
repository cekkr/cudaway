# Linux LD_PRELOAD Packaging Blueprint

This note fulfills the P1 requirement to capture the operational recipe for distributing the
CUDAway LD_PRELOAD shim. It should be kept in sync with `README.md` + `AI_REFERENCE.md` whenever the
build graph or exported symbols change.

## 1. Symbol Inventory

We must ship two shared objects on Linux:

1. `libcuda.so` – exposes the CUDA Driver API (`cu*` surface) plus any math-library entry points that
   some apps expect to resolve from the same binary (e.g., legacy cuBLAS exports). Prefer exporting
   only the driver symbols at first; math libraries will live in dedicated `.so` later.
2. `libcudart.so` – runtime API shim forwarding to the driver layer. Even if most apps preload only
   `libcuda.so`, packaging `libcudart.so` ensures compatibility with binaries that dynamically load
   the runtime instead of relying on libdl stubs.

Symbol definitions live under `src/host/driver` and `src/host/runtime` (future folders). Maintain a
canonical export map:

- Generate `tools/data/cuda_driver_exports.txt` via the existing runtime converter (extend it beyond
  runtime symbols).
- Consume the map from `CMakeLists.txt` using a GNU ld version-script:

```cmake
target_link_options(cudaway_core PRIVATE
    "LINKER:--version-script=${CMAKE_SOURCE_DIR}/cmake/exports/libcuda.version")
```

Version-script template:

```
CUDAWAY_0.1 {
    global:
        cuInit;
        cuDeviceGetCount;
        // ...
    local:
        *;
};
```

## 2. Build Flags & Toolchain Expectations

- Compile all shim sources with `-fPIC -fvisibility=hidden` and only mark exported functions as
  `__attribute__((visibility("default")))` (or via `CMAKE_CXX_VISIBILITY_PRESET`).
- Link `libcuda.so` and `libcudart.so` with `-Wl,-E -ldl -pthread` to mirror NVIDIA's loader
  requirements.
- Enable hardening flags by default: `-Wl,-z,defs -Wl,-z,relro -Wl,-z,now`.
- Provide reproducible builds: set `CMAKE_AR`/`CMAKE_RANLIB` to `llvm-ar`/`llvm-ranlib` and bake
  `SOURCE_DATE_EPOCH` into CMake configure environment when producing release artifacts.
- Ship a `cmake/exports/` directory that contains version scripts + generated symbol lists so CI can
  diff export changes.

## 3. Packaging Layout

Recommended artifact structure per release:

```
cudaWay-linux-x86_64/
├── libcuda.so          # Driver shim
├── libcudart.so        # Runtime shim
├── libcuda.map         # Export map/version script snapshot
├── docs/
│   └── LD_PRELOAD.md   # Operational steps (copied from this note)
├── tools/
│   ├── preload-wrapper # Optional helper that sets LD_PRELOAD & HIP vars
│   └── cuda-mapper.json# Generated symbol metadata
└── examples/
    └── smoke_test.sh   # Runs ./build/cudaway against vectorAdd sample
```

Operational steps for users:

1. Install ROCm (or point `CUDAWAY_ROCM_ROOT` at a relocatable ROCm bundle).
2. Copy the `.so` files into a project-local directory (`/opt/cudaway/lib` or similar).
3. Launch target binaries with:
   ```bash
   LD_LIBRARY_PATH=/opt/cudaway/lib:$LD_LIBRARY_PATH \
   LD_PRELOAD=/opt/cudaway/lib/libcuda.so:/opt/cudaway/lib/libcudart.so \
   HIP_PATH=/opt/rocm \
   ./target_binary
   ```
4. For system-wide installs, ship a `/etc/ld.so.conf.d/cudaway.conf` entry pointing at the install
   directory so `ldconfig` caches the shim.

## 4. Validation & Release Ops

- **Symbol diffing:** run `nm -D --defined-only libcuda.so` and compare it against
  `tools/data/cuda_driver_exports.txt` before cutting a release. Fail CI if the diff introduces
  missing symbols or ABI regressions.
- **Integration test:** execute `./build/cudaway` plus a real CUDA sample (vectorAdd, cuBLAS SGEMM)
  with `LD_PRELOAD` set to the newly built shim. Record pass/fail in the release notes.
- **Provenance:** embed `git describe --tags` and ROCm version information via an `.note` ELF
  section (e.g., `objcopy --add-section .note.cudaway=build/metadata.bin libcuda.so`).

## 5. Outstanding Questions

- Do we need split artifacts per ROCm version, or can a single shim negotiate capabilities at
  runtime? (Blocked on the PTX compiler + HIP dependency matrix.)
- How do we ship kernel caches? Current plan: keep them user-local under `$TMPDIR` and rely on warm
  caches rather than distributing binaries.
- Windows packaging (DLL proxy) lives in `studies/ROCm-API-LinuxVsWindows.md`; consolidate both once
  the Linux LD_PRELOAD flow solidifies.
