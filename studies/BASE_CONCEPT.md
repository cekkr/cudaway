# **Project Cerberus: A Technical Blueprint for a Transparent CUDA-on-ROCm Translation Layer for Linux and Windows**

## **Section 1: A High-Level Architecture for CUDA-on-ROCm**

### **1.1 Executive Summary: The Conceptual Model**

This report details the technical architecture for a CUDA-to-ROCm Transparent Translation Layer (CR-TTL). The primary objective of this library is to enable unmodified, binary-only NVIDIA CUDA applications to execute with full transparency on AMD hardware utilizing the Radeon Open Compute (ROCm) software stack. This capability would allow software compiled exclusively for the CUDA ecosystem to run on AMD GPUs without recompilation, code modification, or developer intervention.  
The core architecture is founded on a dual-pronged strategy that separates host-side API calls from device-side kernel execution:

1. **Host API Translation:** A "shim" or "proxy" library that intercepts all CPU-side function calls destined for the NVIDIA CUDA libraries. These calls (e.g., cudaMalloc, cublasSgemm) are captured and semantically translated into their corresponding AMD ROCm/HIP equivalents (e.g., hipMalloc, rocblas\_sgemm).  
2. **Device Kernel Translation:** A Just-in-Time (JIT) compilation pipeline that intercepts the device-side GPU kernels. When a CUDA application attempts to load a kernel, the CR-TTL intercepts the NVIDIA Parallel Thread eXecution (PTX) virtual assembly code. This PTX is then translated, compiled, and optimized at runtime into the native GCN/RDNA instruction set architecture (ISA) for the detected AMD GPU.

This architectural model is not merely theoretical; it has been validated by existing projects. The ZLUDA project, for example, successfully demonstrated this "drop-in" translation layer approach, enabling many real-world CUDA applications to run on non-NVIDIA GPUs by implementing a PTX JIT compiler. The demonstrated success of this precedent validates the technical feasibility of the design presented herein.

### **1.2 Architectural Diagram: The "Two-Layer" Design**

The flow of execution for a CUDA application running under the CR-TTL follows a precise path of interception and translation.

1. A **CUDA Application** (e.g., blender.exe, a PyTorch-backed Python script) initiates a call to the CUDA API, such as cuModuleLoadData to load a kernel or cuLaunchKernel to execute one.  
2. The call is intercepted by the **CR-TTL Host API Layer**. This layer's implementation varies by operating system:  
   * **On Linux:** The CR-TTL takes the form of a single shared object (libcr-ttl.so). The dynamic linker is instructed to load this library before all others by setting the LD\_PRELOAD environment variable.  
   * **On Windows:** The CR-TTL is packaged as a set of proxy Dynamic-Link Libraries (DLLs) named identically to the NVIDIA libraries (e.g., nvcuda.dll). These are placed in the application's directory, causing the OS to load them instead of the system's (absent) NVIDIA driver files.  
3. The Host API Layer's intercepted function (e.g., cuModuleLoadData) retrieves the **PTX Kernel** string from the application's request.  
4. This PTX code is passed to the **CR-TTL Device Kernel JIT** component. This embedded compiler performs a multi-stage translation: PTX Parser \-\> LLVM-IR \-\> LLVM AMDGPU Backend \-\> GCN/RDNA Binary.1  
5. The resulting native AMD machine code binary is loaded into the AMD driver using a native **ROCm/HIP Driver** call (e.g., hipModuleLoadData).  
6. The Host API Layer then receives the original cuLaunchKernel call, translates its launch parameters (grid/block dimensions), and executes the now-native AMD kernel using hipModuleLaunchKernel on the target **AMD GPU**.

All subsequent API calls (e.g., for memory management, stream synchronization, or high-level library operations) are similarly intercepted and translated to their ROCm/HIP counterparts, maintaining a stateful mapping of all CUDA objects (contexts, streams, events) to their corresponding HIP objects.

### **1.3 Analysis of the CUDA API Duality (Driver vs. Runtime)**

A foundational element of this architecture is a correct understanding of the two distinct CUDA APIs: the Driver API and the Runtime API.

1. **CUDA Driver API:** This is the low-level, C-based interface exposed by the NVIDIA driver itself (e.g., libcuda.so on Linux, nvcuda.dll on Windows). It provides fine-grained, explicit control over all GPU resources. It is language-independent, deals in cubin (binary) and PTX objects, and requires manual management of contexts (CUcontext) and modules (CUmodule). All function symbols are prefixed with cu....  
2. **CUDA Runtime API:** This is a higher-level, C++-based convenience layer (e.g., libcudart.so). It provides implicit initialization, automatic context management, and simpler syntax (e.g., cudaMalloc). It is the API most developers interact with. All function symbols are prefixed with cuda....

Crucially, the Runtime API is a convenience wrapper that sits on top of the Driver API. When an application calls a Runtime function like cudaLaunchKernel, the libcudart.so library, under the hood, makes a series of calls to the Driver API (libcuda.so) to execute the request, including functions like cuLaunchKernel.  
This relationship dictates the CR-TTL's primary interception target. While some applications link against the Runtime API, all CUDA execution *must* ultimately pass through the Driver API. Therefore, the most robust and comprehensive strategy is to intercept the **Driver API**. By providing a complete, 1-to-1 replacement for libcuda.so / nvcuda.dll, the CR-TTL captures all CUDA operations, regardless of whether they originated from a high-level Runtime API call or a low-level Driver API call. The CR-TTL will *also* provide shims for the Runtime API to satisfy the linker, but the internal logic of these shims will simply translate the implicit context management and then call the CR-TTL's *own* internal Driver API implementation.

## **Section 2: The Host API Layer: Platform-Specific Interception**

### **2.1 Vector 1: Linux Interception via LD\_PRELOAD**

On Linux, the interception mechanism is powerful and standardized through the dynamic linker.  
**Mechanism:** The LD\_PRELOAD environment variable provides a list of user-specified shared object files for the dynamic linker (ld.so) to load *before* any other system or application library. This allows the preloaded library to provide its own implementation of function symbols, which will be resolved by the linker in preference to any symbols in subsequently loaded libraries.  
Implementation:  
The CR-TTL will be compiled as a single, monolithic shared library (e.g., libcr-ttl.so). The end-user will launch their unmodified CUDA application by prepending this variable:  
$ export LD\_PRELOAD=/path/to/libcr-ttl.so  
$./my\_cuda\_application  
When the application starts and the linker attempts to resolve a symbol like cuInit, it will find and bind the implementation within libcr-ttl.so. This implementation will *not* be a proxy that forwards the call using dlsym(RTLD\_NEXT,...), as there is no "real" libcuda.so to forward to. Instead, libcr-ttl.so provides the *full* implementation. For example, its cuInit function will directly call hipInit(0).  
To provide full transparency, this single libcr-ttl.so file must export symbols for the *entire* CUDA ecosystem that an application might link against, including:

* **Core:** libcuda.so (Driver API) and libcudart.so (Runtime API)  
* **Math:** libcublas.so (BLAS) and libcusparse.so (Sparse)  
* **AI/DL:** libcudnn.so (Deep Learning)  
* **Signals:** libcufft.so (FFT)  
* **Scale-Out:** libnccl.so (Multi-GPU)  
* **Monitoring:** libnvml.so (Management)

By providing all these symbols in one preloaded library, the CR-TTL can satisfy the linker for any CUDA application and ensure all calls are routed to the ROCm translation backend.

### **2.2 Vector 2: Windows Simulation via DLL Proxying**

Windows lacks a direct LD\_PRELOAD equivalent, necessitating a different strategy known as DLL proxying or hijacking.  
**Mechanism:** This approach leverages the standard Windows DLL Search Order. When an application loads a library, Windows searches a well-defined set of directories. Critically, the directory from which the application was loaded is searched *before* most system directories. An attacker can abuse this by placing a malicious DLL with the same name as a legitimate system DLL in the application directory. For the CR-TTL, this is the core simulation mechanism.  
Implementation:  
The CR-TTL for Windows will be compiled into a series of proxy DLLs, each named identically to the NVIDIA library it replaces:

* nvcuda.dll (Driver API)  
* cudart64\_110.dll (Runtime API, versioned)  
* cublas64\_11.dll (cuBLAS, versioned)  
* cudnn64\_8.dll (cuDNN, versioned)  
* ...and so on for cuFFT, cuSPARSE, nvml, etc.

The user, who would not have the NVIDIA driver installed, will place these proxy DLLs in the same directory as their application's executable (e.g., blender.exe). When the application starts, it will call LoadLibrary("nvcuda.dll"). The Windows loader will find the CR-TTL's nvcuda.dll in the local directory and load it.  
These proxy DLLs will export all the same function names and ordinals as the original NVIDIA libraries. The implementation of nvcuda.dll::cuInit, for example, will call hipInit from the amdhip64.dll (which must be installed as part of the system's ROCm-enabled driver). This "simulation" approach is more complex to maintain than the Linux method, as the CR-TTL must manage and ship dozens of distinct DLL files, each corresponding to the specific file names and versions that different applications are built to find.

## **Section 3: Core Translation: Mapping CUDA (Driver & Runtime) to HIP**

The core logic within the Host API Layer (Section 2\) is a stateful 1-to-1 mapping of CUDA API calls to their ROCm/HIP equivalents. This translation is greatly simplified by the fact that AMD's HIP (Heterogeneous-Compute Interface for Portability) was explicitly designed to be syntactically and semantically similar to CUDA. For many functions, the translation is a simple find-and-replace of the cuda prefix with hip. The primary engineering challenge lies in correctly managing the handles and implicit states between the two ecosystems.

### **3.1 Context and Device Management**

* **CUDA (Driver):** cuInit, cuDeviceGet, cuCtxCreate, cuCtxSetCurrent.  
* **HIP (Runtime):** hipInit, hipGetDevice, hipCtxCreate, hipCtxSetCurrent.

**Implementation:** The CR-TTL must maintain an internal mapping, such as std::map\<CUcontext, hipContext\>. When an application calls cuCtxCreate, the CR-TTL will internally call hipCtxCreate and store the resulting hipContext. It will then return a unique, opaque CUcontext handle (which could be the hipContext pointer itself, type-cast) to the application.  
When the application makes a subsequent call, such as cuCtxSetCurrent(my\_cu\_ctx), the CR-TTL will look up my\_cu\_ctx in its map, retrieve the corresponding hipContext, and call hipCtxSetCurrent(my\_hip\_ctx). This mapping provides a transparent abstraction.  
For the CUDA Runtime API's "primary context", which is implicit and per-thread, the mapping is even simpler. HIP shares this concept, and calls like cudaDeviceSynchronize can be directly translated to hipDeviceSynchronize, which will operate on the same implicit, per-thread context.

### **3.2 Memory Management**

* **CUDA:** cudaMalloc, cudaMemcpy, cudaMemset.  
* **HIP:** hipMalloc, hipMemcpy, hipMemset.

**Implementation:** This is a near 1:1 pass-through. The CR-TTL's implementation of cudaMalloc will simply call hipMalloc and return the void\* pointer directly to the application. The application is agnostic to the origin of the device pointer; it only knows its value. Similarly, cudaMemcpy calls are translated by mapping the cudaMemcpyKind enum (e.g., cudaMemcpyHostToDevice) to the equivalent hipMemcpyKind enum (hipMemcpyHostToDevice) and executing the hipMemcpy call.

### **3.3 Stream and Event Management**

* **CUDA:** cudaStream\_t, cudaEvent\_t, cudaStreamCreate, cudaEventRecord, cudaStreamWaitEvent, cudaDeviceSynchronize.  
* **HIP:** hipStream\_t, hipEvent\_t, hipStreamCreate, hipEventRecord, hipStreamWaitEvent, hipDeviceSynchronize.

**Implementation:** This follows the same stateful mapping pattern as contexts. The CR-TTL will maintain internal maps for streams and events (e.g., std::map\<cudaStream\_t, hipStream\_t\>). When an application calls cudaStreamCreate(\&my\_stream), the CR-TTL will call hipStreamCreate(\&hip\_stream), store the mapping, and return its own opaque handle in my\_stream. A subsequent call, such as cudaStreamWaitEvent(my\_stream, my\_event), will trigger the CR-TTL to look up the corresponding hipStream\_t and hipEvent\_t from its maps and then execute hipStreamWaitEvent(hip\_stream, hip\_event).

### **3.4 Module and Kernel Launch (The JIT Handoff)**

This is the most critical function of the Host API Layer, where it interfaces with the Device Kernel JIT (Section 4). The key functions are from the CUDA Driver API.

* cuModuleLoadData(CUmodule\* module, const void\* image): The image is a character string of PTX code.  
* cuModuleGetFunction(CUfunction\* function, CUmodule module, const char\* name)  
* cuLaunchKernel(CUfunction f, unsigned int gridDimX,...)

**Implementation:**

1. On cuModuleLoadData: When this is called, the CR-TTL does not immediately call a HIP function. Instead, it:  
   a. Receives the image (the PTX code string).  
   b. Stores this string internally, associated with a new, opaque CUmodule handle that it generates.  
   c. Triggers the Device Kernel JIT (Section 4\) to compile this PTX string into a native GCN/RDNA binary.  
   d. Once the JIT returns the GCN binary (as a void\* buffer), the CR-TTL loads this native binary into the ROCm driver: hipModuleLoadData(\&hip\_module, gcn\_binary).  
   e. The CR-TTL stores this final mapping: std::map\<CUmodule\_handle, hipModule\_t\> g\_module\_map;.  
2. **On cuModuleGetFunction:** The CR-TTL receives the CUmodule handle, looks up the corresponding hipModule\_t from its map, and calls hipModuleGetFunction(\&hip\_function, hip\_module, name) to retrieve the native function handle.  
3. **On cuLaunchKernel:** The CR-TTL receives the CUfunction handle (which maps to a hipFunction\_t), translates the launch parameters (grid/block dimensions, shared memory, and stream), and calls hipModuleLaunchKernel(hip\_function,...) to execute the native AMD kernel on the GPU.

## **Section 4: The Device Kernel Layer: PTX-to-GCN JIT Compiler**

This embedded JIT compiler is the technological core of the CR-TTL, responsible for translating NVIDIA device code to AMD device code at runtime. This approach was successfully implemented by ZLUDA.2 This is not a "from-scratch" compiler; it is a *pipeline* that embeds and leverages the existing LLVM compiler infrastructure.  
The pipeline is PTX \-\> PTX Parser \-\> LLVM-IR \-\> LLVM-JIT-with-AMDGPU-Backend \-\> GCN ISA. This is possible because AMD's ROCm stack (specifically HIP-Clang) is already built on LLVM, and LLVM includes a mature AMDGPU backend capable of generating optimized GCN/RDNA ISA from LLVM-IR. The problem is therefore reduced from "compiling PTX to GCN" to "translating PTX to LLVM-IR." Historical projects like PLANG provide the blueprint for this exact translation.1

### **4.1 Phase 1: PTX Ingestion and Parsing**

The JIT process begins when the cuModuleLoadData intercept (Section 3.4) passes a string of PTX code.  
**Requirement:** A robust parser must be used to convert this string into an in-memory Abstract Syntax Tree (AST) that represents the PTX program.  
**Solution:** An open-source, permissively licensed PTX parsing library, such as ptx-parser or pyxis-roc/ptxparser, will be integrated into the CR-TTL. While NVIDIA provides its own nvPTXCompiler library, its use is likely prohibited for this purpose by the CUDA EULA, which restricts translation to non-NVIDIA platforms. An independent, open-source parser based on the public PTX specification is the only legally and technically sound path.

### **4.2 Phase 2: Semantic Translation to LLVM-IR**

This phase involves "walking" the PTX AST (from Phase 1\) and emitting a semantically equivalent LLVM-IR module. This translation follows the PLANG model.1

* **Virtual Registers:** PTX uses a large, flat, virtual register file (e.g., .reg.f32 %f;). LLVM-IR is a Static Single Assignment (SSA) form. The most direct translation is to map all PTX virtual registers to alloca (stack allocation) instructions in the LLVM function's entry block. Every PTX instruction is then translated as:  
  1. load from its source alloca(s).  
  2. The core operation (e.g., fadd).  
  3. store to its destination alloca.  
     This "de-optimized" IR will be corrected in Phase 3 by LLVM's standard mem2reg optimization pass, which will convert these memory operations back into efficient SSA-form virtual registers.  
* **Address Spaces:** This mapping is critical for correct memory access.  
  * PTX ld.global / st.global \-\> LLVM load / store using pointer address space 1 (Global Memory).  
  * PTX ld.shared / st.shared \-\> LLVM load / store using pointer address space 3 (Shared/LDS Memory).  
  * PTX ld.local / st.local \-\> LLVM load / store using pointer address space 5 (Local/Private Memory).  
* **Intrinsics and Special Registers:** GPU-specific concepts are mapped to LLVM intrinsics that the AMDGPU backend understands.  
  * PTX %tid.x, %ctaid.x \-\> llvm.amdgcn.workitem.id.x, llvm.amdgcn.workgroup.id.x.  
  * PTX bar.sync \-\> llvm.amdgcn.s.barrier.  
* **Functions:** PTX "device" functions are translated to standard LLVM functions. PTX "kernel" functions, which are entry points callable from the host, are translated to LLVM functions marked with the amdgpu\_kernel calling convention.

### **4.3 Phase 3: Runtime Code Generation via LLVM JIT**

The CR-TTL will embed the core LLVM libraries to function as a JIT compiler.  
**Implementation:**

1. **Initialization:** The JIT will initialize the LLVM components for the AMDGPU target: LLVMInitializeAMDGPUTargetInfo(), LLVMInitializeAMDGPUTarget(), and LLVMInitializeAMDGPUTargetMC().  
2. **Target Machine:** It will create an LLVM TargetMachine. This requires specifying the target "triple" (amdgcn-amd-amdhsa) and the specific CPU/GPU architecture (e.g., gfx906 for Vega 20, gfx1030 for RDNA2). The target CPU will be detected at runtime by querying the HIP driver.  
3. **Optimization:** The LLVM-IR module from Phase 2 is passed to an LLVM PassManager. This runs the standard optimization pipeline, including mem2reg (to fix the register translation from 4.2), instruction combining, and numerous AMDGPU-specific optimizations.  
4. **Code Generation:** The optimized IR is passed to the TargetMachine, which uses the AMDGPU backend to emit a native GCN/RDNA machine code binary.  
5. **Caching:** This resulting binary (as a void\* buffer) is then returned to the Host API Layer (Section 3.4) to be loaded via hipModuleLoadData. This binary is also cached in an in-memory map (std::map\<ptx\_hash, gcn\_binary\>) to ensure that this entire JIT pipeline runs only *once* per unique kernel, per application launch.

## **Section 5: Ecosystem Support: Translating High-Level Libraries**

To achieve full transparency, the CR-TTL must intercept calls to the entire CUDA ecosystem, not just the core runtime. The AMD hipify project, which translates CUDA source code to HIP source code, provides the definitive mapping tables for this effort. The CR-TTL will implement a *binary, runtime* version of these hipify translations.

### **5.1 cuBLAS to rocBLAS**

* **Purpose:** cuBLAS provides GPU-accelerated Basic Linear Algebra Subprograms (BLAS).  
* **Mapping:** The ROCm equivalent is rocBLAS. The HIPIFY documentation provides a direct 1-to-1 mapping of the API.3  
* **Implementation:** The CR-TTL's cublas... shim will create and manage a std::map\<cublasHandle\_t, rocblas\_handle\>. All calls are then passed through after mapping the handle and stream.

**Table 5.1: Key cuBLAS to rocBLAS Function Mapping**

| CUDA Function | ROCm Function | Description |
| :---- | :---- | :---- |
| cublasCreate(cublasHandle\_t\* h) | rocblas\_create\_handle(rocblas\_handle\* h) | Create library context |
| cublasDestroy(cublasHandle\_t h) | rocblas\_destroy\_handle(rocblas\_handle h) | Destroy library context |
| cublasSetStream(cublasHandle\_t h, cudaStream\_t s) | rocblas\_set\_stream(rocblas\_handle h, hipStream\_t s) | Associate stream with handle |
| cublasSgemm(cublasHandle\_t h,...) | rocblas\_sgemm(rocblas\_handle h,...) | Single-precision GEMM |
| cublasGemmEx(cublasHandle\_t h,...) | rocblas\_gemm\_ex(rocblas\_handle h,...) | Mixed-precision GEMM |

### **5.2 cuDNN to MIOpen**

* **Purpose:** cuDNN provides high-performance primitives for deep neural networks, such as convolution, pooling, and activation functions.  
* **Mapping:** The ROCm equivalent is MIOpen.  
* **Implementation:** This is the most complex shim. The APIs are conceptually similar (using "handles" and "descriptors") but have diverged significantly. The CR-TTL must maintain maps for:  
  * cudnnHandle\_t \-\> miopenHandle\_t  
  * cudnnTensorDescriptor\_t \-\> miopenTensorDescriptor\_t  
  * cudnnActivationDescriptor\_t \-\> miopenActivationDescriptor\_t  
  * ...and many others.

**Table 5.2: Key cuDNN to MIOpen Enum/Function Mapping**

| CUDA Enum/Function | ROCm Enum/Function | Description |
| :---- | :---- | :---- |
| cudnnCreate(...) | miopenCreate(...) | Create handle |
| cudnnCreateTensorDescriptor(...) | miopenCreateTensorDescriptor(...) | Create tensor descriptor |
| CUDNN\_ACTIVATION\_RELU | miopenActivationRELU | RELU activation enum 4 |
| CUDNN\_ACTIVATION\_SIGMOID | miopenActivationLOGISTIC | Sigmoid activation enum 4 |
| CUDNN\_ACTIVATION\_IDENTITY | miopenActivationPASTHRU | Identity/Pass-through enum 4 |
| cudnnConvolutionForward(...) | miopenConvolutionForward(...) | Forward convolution |

A critical point of analysis reveals a "documentation trap." Older MIOpen porting guides 5 state that MIOpen *lacks support* for critical features like RNNs, Dropout, and non-FP32 data types. If true, this would make the CR-TTL non-viable for most modern ML workloads. However, a deeper analysis of the *newer*, auto-generated HIPIFY documentation 4 reveals extensive, fine-grained API mapping tables for thousands of CUDNN\_ enums and attributes, including those for the modern cuDNN Graph API. This newer documentation must be considered the "ground truth," indicating that MIOpen's API coverage is vast and the translation is feasible, though highly complex. The old porting guides 5 must be disregarded as obsolete.

### **5.3 cuFFT to rocFFT**

* **Purpose:** cuFFT provides GPU-accelerated Fast Fourier Transforms.  
* **Mapping:** The ROCm equivalent is rocFFT.  
* **Implementation:** Both APIs are "plan-based," meaning a configuration (cufftHandle or rocfft\_plan) is created first, then executed. The CR-TTL will map these plan handles.

**Table 5.3: Key cuFFT to rocFFT Function Mapping**

| CUDA Function | ROCm Function | Description |
| :---- | :---- | :---- |
| cufftPlanMany(...) | rocfft\_plan\_create(...) | Create an FFT plan |
| cufftDestroy(cufftHandle p) | rocfft\_plan\_destroy(rocfft\_plan p) | Destroy plan |
| cufftExecC2C(cufftHandle p,...) | rocfft\_execute(rocfft\_plan p,...) | Execute plan (Complex-to-Complex) |

### **5.4 cuSPARSE to rocSPARSE**

* **Purpose:** cuSPARSE provides routines for sparse matrix operations.  
* **Mapping:** The ROCm equivalent is rocSPARSE.  
* **Implementation:** Similar to cuBLAS, this shim maps handles (cusparseHandle\_t \-\> rocsparse\_handle) and matrix descriptors (cusparseMatDescr\_t \-\> rocsparse\_mat\_descr). ROCm documentation also mentions hipSPARSE, which is a portability wrapper that calls rocSPARSE or cuSPARSE depending on the platform. The CR-TTL must *not* target hipSPARSE, as this would create a redundant double-translation (CR-TTL \-\> hipSPARSE \-\> rocSPARSE). The shim must map cuSPARSE *directly* to rocSPARSE for maximum control and performance.

**Table 5.4: Key cuSPARSE to rocSPARSE Function Mapping**

| CUDA Function | ROCm Function | Description |
| :---- | :---- | :---- |
| cusparseCreate(cusparseHandle\_t\* h) | rocsparse\_create\_handle(rocsparse\_handle\* h) | Create handle |
| cusparseSetStream(cusparseHandle\_t h,...) | rocsparse\_set\_stream(rocsparse\_handle h,...) | Set stream |
| cusparseSpMV(...) | rocsparse\_spmv(...) | Sparse Matrix-Vector Multiply |

### **5.5 NCCL to RCCL**

* **Purpose:** The NVIDIA Collective Communications Library (NCCL) implements multi-GPU and multi-node communication primitives (e.g., AllReduce, Broadcast).  
* **Mapping:** The ROCm equivalent is RCCL (ROCm Communication Collectives Library).  
* **Implementation:** This is the most trivial translation. AMD designed RCCL to be a *binary-compatible, drop-in replacement* for NCCL. The RCCL library (librccl.so) exports function symbols that are *identically named* to the NCCL API.  
* **Conclusion:** The CR-TTL's libnccl.so shim is a simple pass-through. Its implementation of ncclCommInitRank will simply link to and call ncclCommInitRank from the system's librccl.so. No mapping or translation logic is required.

### **5.6 NVML to RSMI**

* **Purpose:** The NVIDIA Management Library (NVML) is a C-based API for monitoring and managing GPU state (e.g., temperature, power, utilization).  
* **Mapping:** The ROCm equivalent is RSMI (ROCm System Management Interface).  
* **Implementation:** This is a stateless, 1-to-1 conceptual translation. The CR-TTL's nvml... functions will call the corresponding rsmi... functions.

**Table 5.6: Key NVML to RSMI Conceptual Mapping**

| CUDA Function | ROCm Function | Description |
| :---- | :---- | :---- |
| nvmlInit() | rsmi\_init() | Initialize library |
| nvmlDeviceGetHandleByIndex(...) | rsmi\_dev\_get\_handle\_by\_index(...) | Get device handle |
| nvmlDeviceGetTemperature(...) | rsmi\_dev\_temp\_metric\_get(...) | Get core temperature |
| nvmlDeviceGetPowerUsage(...) | rsmi\_dev\_power\_ave\_get(...) | Get power draw |
| nvmlDeviceGetUtilizationRates(...) | rsmi\_dev\_activity\_get(...) | Get GPU/memory utilization |

## **Section 6: The Windows Strategy: Circumventing Official Limitations**

A core part of the query is to "circumvent" the limitations of ROCm on Windows. This is a non-trivial requirement, as the official ROCm support for Windows is fundamentally different from that on Linux.

### **6.1 Analysis of the "Windows Problem"**

On Linux, ROCm is a full-featured compute stack. On Windows, AMD provides a "HIP SDK" for consumer Radeon and Ryzen GPUs, but this is *not* the full ROCm stack. Analysis of the official documentation for this Windows SDK reveals limitations that are catastrophic for the goal of transparent CUDA simulation 6:

1. **No Training Support:** The documentation explicitly states: **"No backward pass support (essential for ML training)."**  
2. **Inference Severely Limited:** **"Only LLM batch sizes of 1 are officially supported."**  
3. **Incomplete Stack:** **"Only Pytorch is supported, not the entire ROCm stack."**  
4. **No Scale-Out:** **"The torch.distributed module is currently not supported."** This makes any multi-GPU support (Section 5.5) impossible.

The conclusion is unavoidable: the official, publicly available ROCm drivers and libraries for Windows are **unfit** for this project. Any attempt to build the CR-TTL on top of this official stack would fail to run the vast majority of real-world CUDA workloads, particularly AI and ML training.

### **6.2 The Circumvention Strategy: Building a "CR-TTL Private ROCm Stack"**

The only viable path forward on Windows is for the CR-TTL to *build and distribute its own private, feature-complete versions of the ROCm libraries*.  
The proxy DLLs (Section 2.2) like cudnn64\_8.dll and cublas64\_11.dll *cannot* call the system-installed MIOpen.dll or rocBLAS.dll, as these will be the "limited" versions. Instead, the CR-TTL's proxy DLLs must either be statically linked against or bundled with private, fully-featured builds of these libraries (e.g., cr\_ttl\_miopen.dll).  
This strategy transforms the CR-TTL project on Windows from a simple *translation layer* into a *full software stack distributor*. It becomes responsible for compiling, patching, and distributing a significant portion of the AMD ROCm ecosystem.

### **6.3 Technical Guide: Building the ROCm Stack on Windows**

This is a major engineering effort. The process involves building the C++ ROCm libraries from source using the HIP-Clang compiler, which is part of the ROCm LLVM toolchain.  
**Prerequisites:**

* Visual Studio 2022 (C++ toolchain) 7  
* CMake 7  
* Ninja build system 7  
* Python 3  
* vcpkg package manager 7

Building MIOpen.dll (cuDNN Equivalent):  
The process for building MIOpen on Windows, as documented in developer forums 7, is illustrative of the complexity:

1. Install dependencies (e.g., bz2, sqlite3, boost) using vcpkg.7  
2. Manually patch the source or environment to add missing POSIX headers (e.g., creating a fake unistd.h).7  
3. Configure the build with a complex CMake command specifying the HIP backend, platform, and numerous flags to disable problematic features (e.g., \-DMIOPEN\_BACKEND=HIP, \-DHIP\_PLATFORM="amd", \-G "Ninja").7

A similar from-source build process using hip-clang must be repeated for rocBLAS, rocFFT, rocSPARSE, RCCL, and RSMI. This is the single greatest technical risk and engineering cost for Windows support. It creates a massive, ongoing maintenance burden, as the entire private ROCm stack must be re-compiled, patched, and validated for every new AMD driver release or ROCm software update.

## **Section 7: Architectural Precedents, Risks, and Conclusions**

### **7.1 Architectural Precedent: ZLUDA**

The ZLUDA project serves as a critical, external proof-of-concept for the entire CR-TTL architecture. ZLUDA successfully implemented the PTX \-\> JIT \-\> AMD pipeline described in Section 4\.2  
The performance results of this approach were exceptionally promising. Benchmarks showed ZLUDA-enabled CUDA applications (like Blender) running on AMD GPUs with performance that not only rivaled but, in some cases, *exceeded* the performance of the same applications that had been *natively ported* to AMD's HIP API.  
This outcome strongly validates the PTX-JIT architecture. It suggests that the PTX code embedded in CUDA binaries is a highly-optimized, stable intermediate representation (IR) that has already benefited from nvcc's advanced-frontend optimizations. Translating this optimized IR at runtime (as ZLUDA and the CR-TTL do) may be more efficient than a naive source-to-source hipify translation of the original C++ code, which would then rely on the ROCm/hip-clang optimizer to re-discover those same optimizations. The new ZLUDA architecture, targeting ROCm 6.1+ and RDNA1+ GPUs, further aligns with the strategy proposed in this report.8

### **7.2 Summary of Implementation Phases (Recommended Roadmap)**

1. **Phase 1 (Core):** Build the Linux LD\_PRELOAD layer (Section 2.1). Implement only the core Driver API (Section 3\) and the PTX-JIT pipeline (Section 4). The goal is to successfully run a simple CUDA Driver API application, such as vectorAddDrv.  
2. **Phase 2 (Runtime & Ecosystem):** Add the Runtime API shims and the library shims for cuBLAS, cuSPARSE, and cuFFT (Section 5).  
3. **Phase 3 (AI/ML):** Implement the complex cuDNN \-\> MIOpen mapping (Section 5.2), using the HIPIFY tables 4 as the specification.  
4. **Phase 4 (Windows):** Begin the Windows port. This involves two parallel efforts: (A) creating the DLL proxy shims and (B) establishing the from-source build environment for the entire ROCm stack (Section 6.3).  
5. **Phase 5 (Scale-out):** Implement the NCCL \-\> RCCL (Section 5.5) and NVML \-\> RSMI (Section 5.6) shims.

### **7.3 Known Risks and Final Conclusion**

This report outlines a technically feasible, though highly complex, path to achieving transparent CUDA-on-ROCm simulation. The architecture is sound and validated by external precedent. However, the project carries several risks of extreme severity.

* **1\. Legal Risk (High):** This is the single greatest threat to the project. NVIDIA's licensing terms, particularly those introduced around CUDA 11.5, appear to explicitly forbid this project. The EULA states: "You may not reverse engineer, decompile or disassemble any portion of the output generated using Software elements for the purpose of translating such output artifacts to target a non-NVIDIA platform". The PTX-JIT compiler (Section 4\) is a *direct and unambiguous* violation of this clause. This is a legal, not technical, risk that likely acts as a "poison pill" for any commercial or public-facing version of this project.  
* **2\. Technical Risk: Windows Stack (High):** The entire Windows strategy (Section 6\) hinges on "circumvention" of the official, limited ROCm SDK.6 This makes the CR-TTL project responsible for building, patching, distributing, and maintaining its own fork of the ROCm library stack.7 This is a massive, ongoing engineering cost and maintenance burden that could easily overwhelm the project.  
* **3\. Technical Risk: JIT Performance (Medium):** While ZLUDA's *execution* performance was high, a JIT-based approach will always introduce a "warm-up" delay at application startup as kernels are compiled for the first time. For applications that load many kernels, this delay could be significant. Aggressive on-disk caching of compiled GCN binaries (keyed by PTX hash) is essential to mitigate this.  
* **4\. Technical Risk: API Gaps (Medium):** The CUDA ecosystem is a rapidly moving target. While the HIPIFY tables show high coverage 4, there will always be a "long tail" of esoteric functions, deprecated features, or brand-new APIs (like CUDA Graphs or cuSPARSELt) that are not yet mapped. This project will be in a constant race to maintain compatibility with the latest CUDA-enabled software.

Conclusion:  
This blueprint provides a viable technical path for achieving CUDA binary compatibility on ROCm. The Linux implementation is well-defined, and the JIT-based kernel translation approach is validated by ZLUDA's high-performance results. The Windows implementation is an order of magnitude more complex, not due to the translation itself, but due to the immaturity of the official ROCm stack on that platform, which necessitates building a private, full-featured ROCm fork.  
The entire endeavor, however, is overshadowed by the clear and present legal risk posed by the NVIDIA EULA. This must be the primary "go/no-go" decision point before any technical work is undertaken.

#### **Bibliografia**

1. “PLANG: Translating NVIDIA PTX language to LLVM IR Machine” \- YouTube, accesso eseguito il giorno novembre 7, 2025, [https://www.youtube.com/watch?v=o-T8Dn8WX9E](https://www.youtube.com/watch?v=o-T8Dn8WX9E)  
2. Comparison to other solutions \- SCALE documentation, accesso eseguito il giorno novembre 7, 2025, [https://docs.scale-lang.com/stable/manual/comparison/](https://docs.scale-lang.com/stable/manual/comparison/)  
3. CUBLAS API supported by HIP and ROC — HIPIFY Documentation, accesso eseguito il giorno novembre 7, 2025, [https://rocm.docs.amd.com/projects/HIPIFY/en/amd-staging/tables/CUBLAS\_API\_supported\_by\_HIP\_and\_ROC.html](https://rocm.docs.amd.com/projects/HIPIFY/en/amd-staging/tables/CUBLAS_API_supported_by_HIP_and_ROC.html)  
4. CUDNN API supported by MIOPEN — HIPIFY Documentation, accesso eseguito il giorno novembre 7, 2025, [https://rocm.docs.amd.com/projects/HIPIFY/en/docs-6.4.0/reference/tables/CUDNN\_API\_supported\_by\_MIOPEN.html](https://rocm.docs.amd.com/projects/HIPIFY/en/docs-6.4.0/reference/tables/CUDNN_API_supported_by_MIOPEN.html)  
5. MIOpen Porting Guide — MIOpen Documentation, accesso eseguito il giorno novembre 7, 2025, [https://rocm.docs.amd.com/projects/MIOpen/en/docs-5.3.3/MIOpen\_Porting\_Guide.html](https://rocm.docs.amd.com/projects/MIOpen/en/docs-5.3.3/MIOpen_Porting_Guide.html)  
6. Ryzen Limitations and recommended settings — Use ROCm on ..., accesso eseguito il giorno novembre 7, 2025, [https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsryz.html](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsryz.html)  
7. How to build MIOpen.dll on Windows · ROCm MIOpen ... \- GitHub, accesso eseguito il giorno novembre 7, 2025, [https://github.com/ROCm/MIOpen/discussions/2703](https://github.com/ROCm/MIOpen/discussions/2703)  
8. ZLUDA \- ZLUDA's third life, accesso eseguito il giorno novembre 7, 2025, [https://vosen.github.io/ZLUDA/blog/zludas-third-life/](https://vosen.github.io/ZLUDA/blog/zludas-third-life/)