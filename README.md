# Custom CUDA Kernel Library for Common ML Operators

CUDA C++ implementations of FP32 MatMul, LayerNorm and row-wise Softmax, with CPU references, correctness tests, shape-specific kernels, validated auto-tuning, and cuBLAS/cuDNN comparisons.

The reference workload uses GPT-2 dimensions: hidden size 768, 12 attention heads, head dimension 64 and intermediate size 3072. This is a reproducible reference workload, not evidence from a team's production models. See [TARGET_WORKLOAD.md](TARGET_WORKLOAD.md).

## Windows prerequisites

- NVIDIA GPU and a compatible driver/toolkit. This machine uses RTX 5050 Laptop (sm_120), driver 591.97 and CUDA Toolkit 13.4.
- Visual Studio 2022 C++ Build Tools and Windows SDK.
- CMake 3.24 or newer; Python 3.8 or newer for analysis.
- cuDNN 9 for normalization vendor comparisons. A checksum-verified CUDA 13 package is installed locally under third_party; it is not committed.
- Optional matplotlib for charts; Nsight Compute and Compute Sanitizer for profiling/validation.

## Build and run

Run from the project root in Command Prompt:

```bat
build.bat
ctest --test-dir build --output-on-failure
build\benchmark_targets.exe --csv results\target_results.csv
python -X utf8 scripts\analyze_results.py --csv results\target_results.csv --plot
```

`run_all.bat` performs those steps and stops when a required step fails. The default architecture is `120-real` for this RTX 5050. To build for another GPU, pass its architecture to build.bat, e.g. `build.bat 86-real`.

Manual configuration in the x64 Native Tools Command Prompt:

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120-real
cmake --build build
```

If a build directory was moved or renamed, configure once with `cmake --fresh` using the same options. The build script detects a moved cache.

To install the pinned local cuDNN dependency on another Windows machine:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_cudnn.ps1
build.bat
```

Alternatively configure with `-DCUDNN_ROOT=C:/path/to/cudnn-archive`. A build without cuDNN remains usable, but the target benchmark exits with status 2 to mark missing vendor comparisons rather than implying completion.

## Target benchmark

Default: 5 independent trials, 50 timing samples per trial, rotated kernel order. All candidate outputs are validated before timing. Allocation, transfers, tuning, graph construction and vendor plan setup are excluded. CUDA graph replay amortizes launch submission overhead; results describe cached-buffer steady-state GPU execution, not end-to-end model latency.

```bat
build\benchmark_targets.exe --quick --trials 3 --samples 20 --csv results\smoke.csv
build\benchmark_targets.exe --matmul 128 64 128 --trials 5 --csv results\attention_matmul.csv
build\benchmark_targets.exe --layernorm 128 768 --csv results\layernorm.csv
build\benchmark_targets.exe --softmax 12288 1024 --csv results\softmax.csv
```

Outputs include the aggregate CSV, a `.trials.csv` file containing individual trials, and `.metadata.json` with hardware and timing settings. Dimensions are positive integers. Multiple workload flags may be supplied.

MatMul compares against cuBLAS SGEMM with pedantic FP32 math, with TF32 conversion disabled by that mode. LayerNorm uses the cuDNN LayerNorm inference graph; Softmax uses cuDNN accurate instance Softmax. The cuDNN engine is selected by vendor heuristics. Library versions and precision are part of the comparison.

## Results and claims

Use [results/summary.md](results/summary.md), generated from actual measurements. A ratio above 1 means custom is faster. The report identifies a vendor win only when a custom kernel is at least 5% faster in every recorded trial. Losses and variable advantages remain in the report. The library does not claim to beat vendor kernels for every shape.

The console ends with a final comparison table showing best custom/kernel latency, vendor latency, speedup, and trial-based result for every target shape. CUSTOM WIN means at least 5% improvement in every trial; CUSTOM LEADS marks an aggregate advantage that has not met that threshold.

Speedup against V1 uses the actual V1 row, not the slowest implementation. Older CSVs remain readable. The analyzer can locate project/build results automatically; explicit --csv always takes precedence.

## Implementations

| Operator | Baselines and variants | Shape-specific implementation |
|---|---|---|
| MatMul | Naive, shared-memory tiles 16/32, register blocking, scalar tiled/unrolled V4, cuBLAS | Cooperative loads, padded shared-memory layout, register patches; split-K reduction for a single input row |
| LayerNorm | Shared-memory reductions, warp/cross-warp reductions, Welford/float4 V4, cuDNN | Values cached in registers; 128-thread resident row kernels for Transformer widths |
| Softmax | Shared-memory reductions, warp reductions, online/float4 V4, cuDNN | Warp-per-row reductions with values retained in registers and explicit fast exponential |

V4 LayerNorm/Softmax fall back safely for unaligned buffers or row widths. Online V4 computes statistics in one pass and rereads input for output; it is not one total input pass. Global fast-math is off by default; the shape Softmax uses __expf explicitly and is checked numerically.

## Public API

```cpp
#include "kernel_dispatcher.cuh"
// Buffers are already on the GPU; row-major, contiguous, distinct inputs/outputs.
cuda_kernels::autotune_matmul(M, K, N, cublas_handle);
cuda_kernels::matmul(d_A, d_B, d_C, M, K, N, cublas_handle);
cuda_kernels::autotune_layernorm(batch, hidden);
cuda_kernels::layernorm(d_x, d_gamma, d_beta, d_y, batch, hidden);
cuda_kernels::autotune_softmax(batch, cols);
cuda_kernels::softmax(d_x, d_y, batch, cols);
```

Tuning uses initialized deterministic inputs, validates candidates, then caches the fastest median. Cache keys include the CUDA device and distinguish MatMul with/without a vendor handle; LayerNorm includes epsilon. Caches are protected by a mutex and last for the process. Host argument errors throw std::invalid_argument; CUDA failures terminate through CUDA_CHECK. Zero rows are a no-op, while feature/reduction dimensions must be positive. Element counts are limited to INT_MAX.

## Tests and profiling

The four CTest suites cover original shapes, odd widths, misaligned buffers, constant rows, extreme finite Softmax inputs, tuned dispatch, invalid arguments and NaN/Inf rejection.

```bat
compute-sanitizer --tool memcheck --error-exitcode 99 build\test_edge_cases.exe
compute-sanitizer --tool racecheck --error-exitcode 99 build\test_edge_cases.exe
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\profile.ps1
```

Profiling reports live under results/profiles. Profiler-instrumented timings are kept there and must not be used as performance evidence. If Nsight reports ERR_NVGPUCTRPERM, enable developer performance-counter access in NVIDIA Control Panel or run as administrator.

Legacy executable cuda_kernel_library still runs the earlier square benchmark suite and API demonstrations. Standalone benchmark_matmul, benchmark_layernorm and benchmark_softmax remain available. The target benchmark is the principal evaluation workflow.

## Scope

This is a single-GPU FP32 forward-operator library. It does not provide model training, backward kernels, PyTorch bindings, batched GEMM, mixed precision, Tensor Core implementations, disk-persistent tuning, or an end-to-end Transformer. Those are future extensions rather than prerequisites for the stated three-operator project.

See [DOCUMENTATION.md](DOCUMENTATION.md) for mathematics, implementation and experimental methodology.
