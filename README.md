# Custom CUDA Kernel Library for Common ML Operators

CUDA C++ implementations of **FP32 matrix multiplication (MatMul), Layer Normalization (LayerNorm), and row-wise Softmax**, with correctness tests, shape-specific kernels, auto-tuning, and comparisons against NVIDIA cuBLAS and cuDNN.

The project evaluates individual operators using a GPT-2-style reference workload and synthetic inputs. It provides measured results for selected shapes; it does not run a complete GPT-2 model or establish performance on a team's production models.

## Contents

- [Problem, objective, and applications](#problem-objective-and-applications)
- [Build and run](#build-and-run)
- [Project workflow](#project-workflow)
- [Dimensions and target workload](#dimensions-and-target-workload)
- [CUDA execution and scheduling](#cuda-execution-and-scheduling)
- [Kernel implementations](#kernel-implementations)
- [Shape versus Dispatcher and auto-tuning](#shape-versus-dispatcher-and-auto-tuning)
- [Correctness and benchmark methodology](#correctness-and-benchmark-methodology)
- [Result files and final comparison table](#result-files-and-final-comparison-table)
- [Public API](#public-api)
- [Project files](#project-files)
- [Tests and profiling](#tests-and-profiling)
- [Demo explanation](#demo-explanation)
- [Troubleshooting](#troubleshooting)
- [Scope and future improvements](#scope-and-future-improvements)
- [References](#references)
- [Manual commands and demo checklist](#manual-commands-and-demo-checklist)

## Problem, objective, and applications

**Problem statement:** General-purpose vendor kernels serve many workloads. For a fixed set of model shapes, their generality can leave opportunities for specialized implementations. Whether a custom kernel actually wins must be measured.

**Objective:** Develop CUDA kernels for MatMul, LayerNorm, and Softmax; validate their outputs; and benchmark them against vendor implementations on documented target shapes.

**Deliverables:** A reusable CUDA static library, several implementations per operator, an explicit auto-tuner and dispatcher, correctness tests, CSV measurements, a final comparison table, optional charts, and profiling scripts.

| Operator | What it does | Typical ML application |
|---|---|---|
| MatMul | Multiplies two matrices | Hidden projections, fully connected layers, attention score calculations |
| LayerNorm | Normalizes features within each row, then applies scale and bias | Transformer feature normalization |
| Softmax | Converts each row of scores into nonnegative probabilities that sum approximately to one | Attention weights and classification probabilities |

These are applications of the operators. The repository benchmarks the operators separately rather than providing a complete attention or training pipeline.

## Build and run

### Requirements

| Requirement | Purpose |
|---|---|
| NVIDIA CUDA-capable GPU | Execute the kernels |
| Compatible NVIDIA driver and CUDA Toolkit | Run CUDA programs and compile `.cu` sources with `nvcc` |
| Visual Studio 2022 C++ Build Tools and Windows SDK | Compile and link the CPU portions on Windows |
| CMake 3.24 or newer | Configure the build; this project uses C++17 and CUDA C++17 |
| Python 3.8 or newer | Analyze CSV results and run analyzer tests |
| cuDNN 9 SDK | Enable LayerNorm and Softmax vendor comparisons |
| Optional matplotlib | Generate the comparison chart |
| Optional Nsight Compute / Compute Sanitizer | Profile kernels / check memory accesses and shared-memory hazards |

The local setup documented for this project uses an **RTX 5050 Laptop GPU**, compute capability **12.0** (`sm_120`), driver **591.97**, CUDA Toolkit **13.4**, and the cuDNN **9.17.1 CUDA 13** package. These are reproduction details, not universal minimum versions. Compiler, loaded runtime, and driver-supported CUDA versions can differ; check `nvcc --version`, `nvidia-smi`, and the run's metadata separately.

### Quick start

Use **Command Prompt (CMD)** from the project root:

~~~bat
cd /d C:\Users\RUTIKA\Desktop\LY\HPC\cuda-kernel-library
chcp 65001
run_all.bat
~~~

`run_all.bat` builds, runs CTest, benchmarks the target workload, and analyzes the results with optional chart generation. It stops if a required step fails. `build.bat` attempts to initialize the installed Visual Studio Build Tools environment.

To run each step manually:

~~~bat
build.bat
ctest --test-dir build --output-on-failure
build\benchmark_targets.exe --trials 5 --samples 50 --csv results\target_results.csv
python -X utf8 scripts\analyze_results.py --csv results\target_results.csv --plot
~~~

The build script defaults to `120-real`, generating GPU machine code for this RTX 5050 architecture. For another GPU, pass its supported architecture, for example `build.bat 86-real` for a compute-capability 8.6 device. Choose the architecture for the actual GPU and installed toolkit.

The custom library is compiled with `-O3`, line information for profiling, and compiler register/resource reporting. `CUDA_KERNELS_FAST_MATH` defaults to `OFF`; enabling it changes numerical behavior and requires fresh correctness and performance evaluation. It is separate from the explicit fast exponential in Shape Softmax.

### cuDNN setup

The setup script downloads the pinned Windows cuDNN 9.17.1 CUDA 13 archive from NVIDIA, verifies its SHA-256 checksum, and extracts it under `third_party`:

~~~bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_cudnn.ps1
build.bat
~~~

CMake discovers the local archive, links cuDNN, and copies its DLLs beside `benchmark_targets.exe`. The downloaded SDK is excluded from Git. To use another installation, add `-DCUDNN_ROOT=C:/path/to/cudnn-archive` to the manual CMake configuration shown in the command checklist.

A build without cuDNN can run custom kernels and cuBLAS, but the target benchmark returns **exit code 2** to indicate incomplete normalization vendor comparisons. Exit code **1** indicates an error; **0** indicates the benchmark completed with its vendor comparisons available. `run_all.bat` stops on either nonzero benchmark status.

## Project workflow

~~~text
Build CPU and GPU code
        |
Run correctness test suites
        |
Choose default or custom operator dimensions
        |
Generate deterministic inputs and copy them to GPU memory
        |
Prepare references, vendor plans, and auto-tuning selections
        |
Validate every benchmark candidate
        |
Warm up and measure repeated GPU execution
        |
Save aggregate CSV, trial CSV, and metadata
        |
Analyze results: compare kernels, write summary, optionally plot
        |
Display the final custom-versus-vendor comparison table
~~~

`benchmark_targets.exe` performs GPU measurements. `scripts/analyze_results.py` reads saved measurements; analyzing a CSV does **not** run the GPU kernels again.

## Dimensions and target workload

All custom inputs and outputs use contiguous, row-major **FP32** arrays: each stored value is a 32-bit `float`. A shape describes array dimensions; a dtype describes the representation of each value. `RTX 5050` is the GPU model name.

### MatMul dimensions: M x K x N

~~~text
A[M x K] multiplied by B[K x N] gives C[M x N]
C[row, col] = sum over q of A[row, q] * B[q, col]
~~~

For `128x768x3072`, 128 input rows with 768 features are multiplied by a `768x3072` matrix, producing 128 rows with 3072 features. `K` is the shared reduction dimension, not an output dimension.

### LayerNorm and Softmax dimensions

| Operator | Shape format | Meaning |
|---|---|---|
| LayerNorm | `ROWS x HIDDEN` | Independently normalize HIDDEN feature values in each row; gamma and beta each have HIDDEN elements |
| Softmax | `ROWS x COLS` | Independently convert COLS scores in each row into probabilities |

The reference uses hidden width **768**, **12** attention heads, head width **64 = 768 / 12**, and intermediate width **3072 = 4 x 768**. The [published GPT-2 configuration](https://huggingface.co/openai-community/gpt2/blob/main/config.json) supplies the reference hidden width and head count; these benchmarks use selected token counts and sequence lengths. See [TARGET_WORKLOAD.md](TARGET_WORKLOAD.md) for the workload assumptions.

| Operator | Default shape | Interpretation |
|---|---|---|
| MatMul | `1x768x768` | One-token hidden projection |
| MatMul | `16x768x768` | Short-token hidden projection |
| MatMul | `128x768x768` | Projection of 128 token vectors |
| MatMul | `128x768x3072` | MLP feature expansion |
| MatMul | `128x3072x768` | MLP feature contraction |
| MatMul | `128x64x128` | One attention head's score matrix; B already contains the required key-transpose layout |
| LayerNorm | `1x768`, `16x768`, `128x768` | Hidden normalization at different row counts |
| LayerNorm | `128x3072` | Intermediate-width stress case; not a standard GPT-2 LayerNorm location |
| Softmax | `12x128` | 12 heads x one query position, with 128 scores per row |
| Softmax | `1536x128` | 12 heads x 128 query positions, with 128 scores per row |
| Softmax | `3072x256` | 12 heads x 256 query positions, with 256 scores per row |
| Softmax | `12288x1024` | 12 heads x 1024 query positions, with 1024 scores per row |

Softmax is unmasked. Scaling attention scores, causal masking, preparing/transposing keys, and full attention execution are outside the measurements.

### Running different dimensions

Dimensions are runtime arguments; changing them does not require rebuilding:

~~~bat
build\benchmark_targets.exe --matmul 256 512 256 --layernorm 256 512 --softmax 256 512 --trials 5 --samples 50 --csv results\custom_results.csv
python -X utf8 scripts\analyze_results.py --csv results\custom_results.csv --out-dir results\custom --plot
~~~

Multiple shape flags can be supplied, including repeated flags for one operator. Once a shape flag is supplied, the benchmark runs only the explicitly requested cases. Benchmark dimensions must be positive and within the implementation's indexing and GPU-memory limits. A winning result on one shape does not establish a win on another.

## CUDA execution and scheduling

**Compilation runs on the CPU; kernels execute on the GPU.** CMake coordinates the build, `nvcc` compiles CUDA sources, and the Microsoft host compiler handles CPU C++ code. The executable contains CPU code and compiled GPU code. [NVIDIA compiler documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

At runtime the CPU allocates buffers, submits kernel launches, and processes reports. GPU threads execute MatMul, LayerNorm, and Softmax. Helpers in [include/cuda_utils.cuh](include/cuda_utils.cuh) wrap `cudaMalloc`, `cudaMemcpy`, `cudaFree`, CUDA events, streams, and error checks.

| CUDA term | Meaning in this project |
|---|---|
| Thread | Calculates one output value or a small group of values |
| Block | Group of threads that can cooperate through shared memory and block barriers |
| Grid | Collection of blocks launched for a kernel |
| Warp | Group of 32 threads used by GPU execution and warp primitives |
| Streaming Multiprocessor (SM) | GPU hardware unit that executes resident blocks and their warps |
| Global memory | GPU storage for input and output arrays |
| Shared memory | On-chip storage shared by threads within a block |
| Registers | Private thread working storage; actual allocation depends on the compiler |
| Reduction | Combining partial values into a sum, maximum, or another aggregate |
| Stream | Ordered queue of CUDA work; benchmark timing uses a dedicated stream |

A launch such as `kernel<<<grid, block, 0, execution_stream()>>>(...)` specifies work geometry, dynamic shared-memory bytes, and the stream. NVIDIA GPU scheduling assigns blocks to SMs, and hardware warp schedulers issue instructions from ready warps. The project relies on this built-in scheduling; it does not implement a custom hardware or multi-job GPU scheduler. The [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) describes kernel launches and thread organization.

The **Dispatcher selects an implementation**. It does not assign threads to physical GPU cores. Block sizes, shared memory, and register use affect how much work can reside on the GPU, but more threads or higher occupancy do not automatically imply better performance.

## Kernel implementations

The versions are alternative strategies, not a ranking. Every candidate must pass correctness checks before its benchmark timings are used.

### MatMul

Source: [src/matmul.cu](src/matmul.cu).

| Implementation | How it works | Intended benefit |
|---|---|---|
| `Naive(V1)` | One thread calculates one output by reading an A row and B column | Simple reference GPU implementation |
| `Tiled16(V2)`, `Tiled32(V2)` | Cooperatively load input tiles into shared memory, synchronize, and reuse them | Reduce redundant input loads |
| `RegBlock(V3)` | Each thread accumulates a 4x4 output patch; each block covers 32x32 outputs with K chunks of 16 | Reuse inputs across several outputs per thread |
| `Unrolled32(V4)` | Scalar 32x32 shared-memory tiles with groups of four multiply-adds in the inner loop | Reduce loop overhead and expose computation |

In V1, `row = blockIdx.y * blockDim.y + threadIdx.y` and the analogous column expression assign each thread its output location. Flat row-major indexing uses `A[row*K + q]`, `B[q*N + col]`, and `C[row*N + col]`. Bounds checks protect edge threads. In tiled kernels, `__syncthreads()` ensures a tile is loaded before use and not overwritten before all readers finish.

V4 MatMul currently uses **scalar** loads. Some legacy comments and executable labels refer to vectorization; the actual `float4` MatMul loads are in the additional `Vec...` configurations below.

### LayerNorm

Source: [src/layernorm.cu](src/layernorm.cu). For each row:

~~~text
mean = sum(x) / HIDDEN
variance = sum((x - mean)^2) / HIDDEN
y[i] = gamma[i] * (x[i] - mean) / sqrt(variance + epsilon) + beta[i]
~~~

Gamma scales, beta shifts, and positive epsilon stabilizes division. The default epsilon is `1e-5`.

| Implementation | How it works |
|---|---|
| V1 | One 256-thread block per row; shared-memory tree reductions for mean and variance, then normalization |
| V2 | Warp shuffle sums followed by shared-memory communication across warps for both statistics |
| V3 | Similar warp/cross-warp strategy through a reusable block-reduction helper; still uses shared memory between warps |
| V4 | Welford running mean/variance with vectorized four-float accesses when alignment and row width permit |

V1-V3 read the input separately for mean, variance, and output. V4 combines mean and variance accumulation into one input pass, then rereads input to produce output. It does not complete the whole operation with one input read. Unsupported V4 alignment/width conditions fall back to V3.

### Softmax

Source: [src/softmax.cu](src/softmax.cu). For each row:

~~~text
maximum = max(x)
e[i] = exp(x[i] - maximum)
y[i] = e[i] / sum(e)
~~~

Subtracting the maximum avoids exponentiating large positive scores. Correct outputs are nonnegative and sum approximately to one per row.

| Implementation | How it works |
|---|---|
| V1 | Shared-memory tree reductions for maximum and sum; output buffer temporarily stores exponentials |
| V2 | Warp/cross-warp maximum reduction; shared-memory tree for the sum |
| V3 | Warp/cross-warp reductions for both maximum and sum |
| V4 | Online maximum and rescaled exponential sum, combined during the statistics pass; vectorized accesses when safe |

V4 rereads the input to write final probabilities and falls back to V3 for unsupported alignment/width. Global `--use_fast_math` is off by default; the Shape Softmax explicitly uses the approximate `__expf` intrinsic and is validated numerically.

### Shape-specific implementations

Source: [src/shape_kernels.cu](src/shape_kernels.cu). These implementations adjust thread organization and compile-time storage sizes to ranges of dimensions rather than being restricted to the default benchmark shapes.

| Operator / condition | Shape strategy |
|---|---|
| MatMul, `M = 1` | Vector-matrix path: eight warps partition K, then combine partial sums inside one block for each group of output columns |
| MatMul, `2 <= M <= 16` | 16x64 output tiles, K chunks of 32, four outputs per thread |
| MatMul, `M > 16` | 64x64 output tiles, K chunks of 16, a 4x4 output patch per thread |
| LayerNorm, width <= 256 | One warp per row; four independent rows per 128-thread block |
| LayerNorm, width 257-4096 | One 128-thread block per row, retaining input values for statistics and output |
| Softmax, width <= 4096 | One warp per row; four rows per block; retained values and warp maximum/sum |
| LayerNorm / Softmax, width > 4096 | General V3 fallback |

For `LayerNorm 128x768`, there are 128 row blocks; each of 128 threads handles six values. For `Softmax 12288x1024`, each warp handles one row and each lane handles 32 values. Retained arrays are intended for registers, but register allocation and possible spilling must be checked in compiler/profiler output.

Additional MatMul candidates are `Vec32x32_K32`, `Vec32x64_K32`, `Vec32x64_2x4`, `Vec64x64_K32`, `Vec64x64_K64`, `Vec16x32_K32`, `Vec16x64_K32`, and `Vec32x32_K64`. These vary output tile size, K depth, and output work per thread. Four-float input loads require K and N divisible by four and 16-byte-aligned A/B pointers; otherwise a scalar load path is used. `Gemv4`, `Gemv16`, and `Gemv32` partition K among 4, 16, or 32 warps for `M = 1` only. This is an in-block reduction, not a separate global split-K workspace.

## Shape versus Dispatcher and auto-tuning

| Label | Meaning |
|---|---|
| V1-V4 | Individual algorithm implementations for the same operator |
| Shape | Direct call to the shape-specific implementation family |
| Dispatcher | Public API that selects which implementation to call |
| cuBLAS / cuDNN | NVIDIA vendor implementations used for comparison |

When explicitly requested, the auto-tuner generates deterministic inputs, validates candidates, and measures each valid candidate over **three timing trials**, with **10 warm-ups and 30 samples per trial**. It selects the lowest median-of-trial-medians timing and stores the implementation ID in a mutex-protected, process-local cache.

Cache keys include the operator, dimensions, and CUDA device. MatMul keys distinguish calls with and without a cuBLAS handle. LayerNorm includes epsilon's exact FP32 bits; its tuner currently tunes the default `1e-5` epsilon. The cache is lost when the program exits.

Tuning is explicit: calling a dispatcher does **not** benchmark all candidates automatically. With no cached selection, MatMul uses Shape for M <= 16 and V3 otherwise, except that a supplied cuBLAS handle permits a large-workload vendor fallback. LayerNorm and Softmax use Shape up to width 4096 and V3 above it.

The principal target benchmark tunes MatMul **without** a vendor handle, keeping its `Dispatcher` candidate custom-only. An API user who supplies a cuBLAS handle during MatMul tuning allows cuBLAS to compete as a candidate. Norm tuners select among custom implementations only. Shape and Dispatcher may therefore execute the same underlying kernel; small differences in their separately measured times are possible.

## Correctness and benchmark methodology

### Reference outputs and validation

- MatMul compares every candidate against cuBLAS output. For cases with `M*K*N <= 20,000,000`, it also checks cuBLAS against an independent CPU result.
- LayerNorm and Softmax compare all candidates, including the vendor candidate, against CPU references.
- Target candidate tolerance is `abs(output - reference) <= 1e-4 + 1e-4 * abs(reference)`. The small-case cuBLAS-versus-CPU check uses absolute tolerance `1e-3` plus relative tolerance `1e-4`.
- Nonfinite values fail the correctness comparison. Softmax also checks nonnegative outputs and row sums within `1e-4` of one. Standalone test suites use their own documented tolerances.

### How time is measured and stored

Timing is implemented in [include/benchmark.cuh](include/benchmark.cuh) and [benchmarks/benchmark_targets.cu](benchmarks/benchmark_targets.cu).

1. Run 20 warm-up launches on a dedicated CUDA stream.
2. Use a pilot measurement to choose repeated launches per sample, capped at 64.
3. Capture those launches in a CUDA graph, instantiate it, and perform an untimed replay.
4. Record CUDA start/stop events around each graph replay and wait for completion.
5. Use `cudaEventElapsedTime()` and divide by the number of captured operator launches to obtain milliseconds per invocation.
6. Store sample times temporarily in `std::vector<float> times`; calculate average, minimum, and median.
7. Repeat the evaluation over trials, rotating candidate order between trials.

The default target evaluation uses **5 trials x 50 samples** per candidate. A sample can contain multiple operator invocations. The event interval includes replayed GPU work and inter-launch execution effects; it is not an isolated instruction timer.

Aggregate `avg_ms` is the mean of trial averages; `min_ms` is the smallest trial minimum. For five trials, `med_ms` is the median of the five trial medians. If an even trial count is requested, the current aggregation selects the upper middle trial median. Raw individual sample times are not saved to CSV; per-trial statistics are saved.

**Excluded from the performance numbers:** allocation, host/device transfers, CPU references, correctness checks, tuning, graph construction/instantiation, and vendor plan setup. Reused buffers may be cache-resident. Results describe cached-buffer GPU execution, not complete application latency, cold-memory performance, or host launch latency. Rotating order reduces a fixed ordering bias but does not eliminate clock, power, or thermal variation.

### Vendor baselines and precision

| Operator | Principal target baseline |
|---|---|
| MatMul | cuBLAS SGEMM with `CUBLAS_PEDANTIC_MATH`; row-major data is handled by swapping operands/dimensions in the call, without an explicit transpose kernel |
| LayerNorm | cuDNN LayerNorm inference operation graph with a supported plan selected from vendor heuristics |
| Softmax | cuDNN `CUDNN_SOFTMAX_ACCURATE`, `CUDNN_SOFTMAX_MODE_INSTANCE` |

The custom kernels use FP32 and do not implement Tensor Core paths. Pedantic cuBLAS math disables the TF32 acceleration used in relaxed modes and may be slower than other vendor configurations. These results therefore do not establish a win against the fastest possible cuBLAS/cuBLASLt configuration or other dtypes. [cuBLAS math-mode documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t)

## Result files and final comparison table

For `--csv results\target_results.csv`, the workflow produces:

| File | Contents |
|---|---|
| [results/target_results.csv](results/target_results.csv) | Aggregate timing statistics for each operator, shape, and implementation |
| [results/target_results.trials.csv](results/target_results.trials.csv) | Individual trial averages, minima, medians, and throughput |
| `results/target_results.metadata.json` | GPU, compute capability, runtime/driver API versions, cuDNN label, precision, seeds, and timing settings |
| [results/summary.md](results/summary.md) | Human-readable comparison generated by the analyzer |
| `results/benchmark_chart.png` | Optional chart, generated with `--plot` when matplotlib is installed |

Metadata does not currently record every reproduction detail, such as GPU clocks, power state, or the exact cuBLAS build version. Record those separately when making a formal performance claim. Old validation reports or profiler runs may describe a different dataset; use artifacts from the same benchmark run together.

The aggregate CSV fields are:

| Field | Meaning |
|---|---|
| `operator`, `shape`, `kernel` | Operation, dimensions, and measured implementation |
| `avg_ms`, `min_ms`, `med_ms` | Aggregate average, minimum, and trial-median timing, in milliseconds |
| `tflops` | MatMul throughput: `2*M*K*N / (avg_ms * 1e9)`; zero for norms because no FLOP count is supplied |
| `speedup_vs_naive` | V1 aggregate average divided by this implementation's aggregate average |
| `vendor_speedup_median` | Vendor aggregate median divided by this implementation's aggregate median |
| `vendor_speedup_min_trial` | Smallest vendor/custom median ratio across corresponding trials |
| `trial_count` | Number of recorded timing trials |

Zero `tflops` for LayerNorm or Softmax does not mean zero computation. Latency is the principal comparison metric for all three operators.

### Analysis workflow

~~~text
Read selected CSV -> validate timing fields -> group by operator and shape
-> find fastest custom row -> compare with vendor -> write report
-> optionally generate chart -> print final comparison table
~~~

The analyzer defaults to `med_ms`; `--metric avg_ms` selects averages. It uses the actual V1 row for the V1 comparison. CSV `speedup_vs_naive` values from the target executable use averages; the analyzer recomputes ratios using its selected metric. Missing V1 or vendor rows are reported rather than invented. Explicit `--csv` takes precedence over automatic project/build CSV discovery.

### Reading the final table

| Column | Meaning |
|---|---|
| Operator / Shape | Operation and input dimensions |
| Best custom | Custom candidate with the lowest aggregate latency under the selected metric |
| Custom ms / Vendor ms | Execution latency in milliseconds; lower is better |
| Vendor | cuBLAS for MatMul or cuDNN for normalization operators |
| Speedup | Vendor latency divided by best-custom latency |
| Result | Trial-based classification below |

`2.0x` means the custom time is half the vendor time. `0.5x` means the custom time is twice the vendor time. A `1.25x` speedup corresponds to 20% lower latency: `1 - 1/1.25`.

| Result label | Rule |
|---|---|
| `CUSTOM WIN` | Median metric, aggregate speedup >= 1.05, at least three trials, and speedup >= 1.05 in every recorded trial |
| `CUSTOM LEADS` | Aggregate custom latency is lower, but the confirmed-win criterion is not met |
| `VENDOR LEADS` | Aggregate vendor latency is lower |
| `TIE` | The selected aggregate timings are equal |
| `NO VENDOR` | No vendor result is available for this shape |

The repeatability criterion is a project rule, not a statistical confidence interval. Average-mode analysis does not certify a median-based `CUSTOM WIN`. Legacy CSVs without trial evidence cannot establish that classification.

The final confirmed-win count and the fastest implementation can change between runs. Use the current CSV and generated table rather than quoting an earlier count. Improvements against V1 and improvements against a vendor are separate claims. Attributing a gain to one individual optimization requires controlled comparisons with that optimization changed independently.

## Public API

The library target is `cuda_kernels_lib`. These calls assume the caller has already allocated and initialized distinct, contiguous GPU buffers of the required sizes:

~~~cpp
#include "kernel_dispatcher.cuh"

// A[M*K], B[K*N], C[M*N]; custom-only selection when no vendor handle is passed.
cuda_kernels::autotune_matmul(M, K, N);
cuda_kernels::matmul(d_A, d_B, d_C, M, K, N);

// x/y[rows*hidden], gamma/beta[hidden]; tune default epsilon 1e-5.
cuda_kernels::autotune_layernorm(rows, hidden);
cuda_kernels::layernorm(d_x, d_gamma, d_beta, d_y, rows, hidden);

// x/y[rows*cols].
cuda_kernels::autotune_softmax(rows, cols);
cuda_kernels::softmax(d_scores, d_probabilities, rows, cols);
~~~

Tune once during initialization, outside a timed workload. To allow vendor MatMul selection, supply the same valid `cublasHandle_t` to both `autotune_matmul` and `matmul`; it uses a separate cache key. Direct `launch_*` functions are declared in each operator's header and in `shape_kernels.cuh`.

The public API uses the execution stream selected by `set_execution_stream()` from `cuda_utils.cuh`, or the default stream when none is selected. Kernel launches are generally asynchronous; synchronize the relevant stream before the host consumes results or releases buffers still in use. The dispatcher cache mutex does not make shared buffers or a shared vendor handle safe for arbitrary concurrent use.

Argument checks reject negative row counts, nonpositive feature/reduction dimensions, null buffers for nonempty operations, and element counts exceeding `INT_MAX`. LayerNorm epsilon must be finite and positive. Zero rows are a no-op in the library API; the benchmark CLI requires positive rows. Argument failures throw `std::invalid_argument`; `CUDA_CHECK` / `CUBLAS_CHECK` print failures and terminate. Correctness checks reject NaN/Inf comparisons; the kernel API does not scan all inputs for nonfinite values.

## Project files

| Path | Responsibility |
|---|---|
| [CMakeLists.txt](CMakeLists.txt) | Library/executable targets, dependencies, architecture and compiler options, CTest registration |
| [build.bat](build.bat) | Set up Windows compiler environment, configure, and build |
| [run_all.bat](run_all.bat) | Build -> test -> target benchmark -> analysis |
| [src/matmul.cu](src/matmul.cu) | CPU MatMul reference, V1-V4, cuBLAS wrapper |
| [src/layernorm.cu](src/layernorm.cu), [src/softmax.cu](src/softmax.cu) | CPU references and V1-V4 for normalization operators |
| [src/shape_kernels.cu](src/shape_kernels.cu) | Shape-specific kernels and additional MatMul configurations |
| [src/kernel_dispatcher.cu](src/kernel_dispatcher.cu) | Explicit auto-tuning, cache, and public dispatch functions |
| [src/vendor_baselines.cpp](src/vendor_baselines.cpp) | cuDNN LayerNorm plan and Softmax wrapper |
| [include/benchmark.cuh](include/benchmark.cuh) | Timing, statistics, correctness comparisons, CSV helpers |
| [include/cuda_utils.cuh](include/cuda_utils.cuh), [include/validation.cuh](include/validation.cuh) | Memory/stream/event helpers and argument checks |
| [benchmarks/benchmark_targets.cu](benchmarks/benchmark_targets.cu) | Principal target workload, CLI, validation, trials, and result files |
| `benchmarks/benchmark_*.cu` | Older individual-operator timing executables |
| [src/main.cu](src/main.cu) | Legacy full-suite executable and API demonstrations |
| `tests/` | Three operator suites, edge-case suite, and Python analyzer regression tests |
| [scripts/analyze_results.py](scripts/analyze_results.py) | CSV analysis, Markdown report, optional chart, final comparison table |
| [scripts/profile.ps1](scripts/profile.ps1), [scripts/setup_cudnn.ps1](scripts/setup_cudnn.ps1) | Representative Nsight profiles and pinned cuDNN installation |
| [TARGET_WORKLOAD.md](TARGET_WORKLOAD.md), [DOCUMENTATION.md](DOCUMENTATION.md) | Workload assumptions and further implementation/methodology detail |
| `results/`, `build/`, `third_party/` | Measurements/reports, generated build files, and local SDK downloads respectively |

## Tests and profiling

CTest registers four GPU suites: `test_matmul`, `test_layernorm`, `test_softmax`, and `test_edge_cases`. If Python is found at configuration time, `test_analysis` adds a fifth registered suite. Tests cover operator correctness, odd widths, offset norm buffers, constant rows, extreme finite Softmax scores, dispatcher paths, invalid arguments, zero-row API calls, and nonfinite comparison rejection. Edge-case output deliberately includes expected NaN/Inf comparison failures before its final pass message.

`compute-sanitizer --tool memcheck` checks memory accesses; `--tool racecheck` checks shared-memory hazards. Nsight Compute profiles registers, occupancy, memory throughput, and execution behavior. Low utilization alone does not identify a unique bottleneck.

`scripts/profile.ps1` profiles one representative custom kernel for each operator: Shape MatMul `128x768x768`, resident LayerNorm `128x768`, and Shape Softmax `12288x1024`. These fixed profiles do not automatically cover every benchmark winner. Reports and instrumented CSVs go under `results/profiles`; use ordinary uninstrumented trials for speedup claims. Commands are in the checklist below.

The legacy executable and standalone timing programs remain available, but they do not all use the principal target benchmark's validation, vendor configuration, and repeated-trial protocol. In particular, the standalone norm benchmarks do not provide cuDNN comparisons. Use `benchmark_targets.exe` for the documented evaluation.

## Demo explanation

1. Show the project title and objective: investigate custom kernels for selected ML operator shapes.
2. Show `TARGET_WORKLOAD.md` and explain M/K/N versus rows/features. Identify the inputs as synthetic.
3. Run correctness tests and show their actual outcome.
4. Run the three-operator demo or a custom shape from the command checklist.
5. Point to the final table: identify the operation, shape, custom time, vendor time, speedup, and evidence label.
6. Explain one custom win or lead and one vendor-leading case using the displayed results.
7. Open the relevant kernel, then explain Shape versus Dispatcher.
8. State FP32, cached-buffer GPU timing, excluded setup/transfers, and the vendor math mode.

Suggested spoken explanation:

> My project implements MatMul, LayerNorm, and Softmax using CUDA. I first check that each implementation produces correct outputs. Then I compare GPU execution time against cuBLAS and cuDNN for selected GPT-2-style shapes. Shape kernels adapt the thread organization to input dimensions, and our auto-tuner measures which implementation works best. The final table shows where custom kernels help and where the vendor remains faster. These are individual-operator FP32 measurements, not full-model latency results.

Identify saved results as saved results when displaying them after a live demo. Avoid promising a particular winner before the current measurements finish.

## Troubleshooting

| Symptom | Explanation and action |
|---|---|
| CMake says the source has no `CMakeLists.txt` | From the project root use `cmake -S . -B build ...`. `-S ..` selects the parent directory. From inside `build`, the source is `..`; the checklist consistently starts at the root. |
| CMake cache refers to a moved build directory | Reconfigure with `cmake --fresh` and the intended generator/architecture. `build.bat` detects a moved cache. Renaming a built directory can invalidate its CMake paths. |
| `cl` or NMake is unavailable | Use an x64 Native Tools Command Prompt or initialize the Visual Studio environment as shown below. Install the C++ Build Tools if missing. |
| Linker cannot open an `.exe`, for example LNK1104 | Close that running test/benchmark, then rebuild. |
| Unsupported PTX toolchain or no suitable kernel image | Check the actual GPU architecture, driver, toolkit, and build cache. For this GPU rebuild with `120-real`; this avoids relying on a newer PTX JIT for these kernels, but compatible runtime/driver versions are still required. |
| cuDNN missing / benchmark exits with status 2 | Install the matching SDK, reconfigure/rebuild, and confirm CMake reports cuDNN enabled. If no LayerNorm plan supports a requested custom shape, that comparison remains incomplete. |
| Analyzer says CSV not found | Run the benchmark successfully first, then pass the exact output filename with `--csv`. |
| Console borders show garbled characters | Run `chcp 65001` in CMD and use a Unicode-capable terminal/font. Python commands use `-X utf8`. |
| No chart appears | Add `--plot`. If needed install the optional dependency with `python -m pip install matplotlib`; CSV and Markdown analysis work without it. |
| Nsight returns `ERR_NVGPUCTRPERM` | Enable developer settings in NVIDIA Control Panel, then Developer -> Manage GPU Performance Counters -> allow access -> Apply, subject to machine policy. |
| Timings or winners change between runs | Check clocks/power/thermal conditions and background GPU work. Rerun full trials and keep aggregate CSV, trial CSV, and metadata together. |

The `.codex-backup` directory, if present, is an edit backup and is not a build/runtime dependency. Build outputs and local SDK downloads are ignored by Git. The default target CSV and trial CSV are explicitly included by `.gitignore`; other experiment CSVs are normally ignored, so retain/share their evidence deliberately.

## Scope and future improvements

Implemented scope: one GPU, contiguous FP32 forward operators, multiple kernels, explicit process-local tuning, correctness checks, measured vendor comparisons, and representative profiling.

Potential next experiments include better tile/thread selection, pipelined MatMul loading, vectorized resident LayerNorm access, and reduced synchronization overhead where correctness permits. These are proposals, not demonstrated gains. Isolate each change, validate it, and repeat the benchmark before making a claim.

FP16/BF16/INT8, Tensor Core paths, backward kernels, model training, batched GEMM, arbitrary tensor strides, causal/fused attention, residual/normalization fusion, PyTorch bindings, multiple GPUs, and disk-persistent tuning are outside the current implementation. New precision or fused-operator experiments require matching vendor functionality and accuracy settings. There is no guarantee of outperforming every vendor mode, shape, or GPU.

## References

- [NVIDIA CUDA compiler documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) explains host/device compilation.
- [NVIDIA CUDA C++ introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) explains memory management, kernels, and launch geometry.
- [NVIDIA cuBLAS math modes](https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t) describes the precision/performance scope of the baseline.
- [GPT-2 configuration](https://huggingface.co/openai-community/gpt2/blob/main/config.json) supplies the reference hidden width and head count.
- [TARGET_WORKLOAD.md](TARGET_WORKLOAD.md) documents the benchmark shape interpretation; [DOCUMENTATION.md](DOCUMENTATION.md) provides additional implementation detail.

## Manual commands and demo checklist

Use Command Prompt (CMD). Run commands from the project root. Close running test or benchmark executables before rebuilding. Choose the benchmark section you need; the commands below include alternative workflows.

### 1. Prepare the terminal and check the environment

~~~bat
cd /d C:\Users\RUTIKA\Desktop\LY\HPC\cuda-kernel-library
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
chcp 65001
nvidia-smi
nvcc --version
cmake --version
python --version
~~~

The compiler setup command is optional if the terminal is already an x64 Native Tools Command Prompt. UTF-8 code page 65001 displays the Unicode borders correctly.

### 2. Configure and build

~~~bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120-real
cmake --build build
~~~

The build-script alternative is:

~~~bat
build.bat
~~~

If CMake reports a cache-directory mismatch after moving or renaming the build folder, reconfigure once with:

~~~bat
cmake --fresh -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120-real
cmake --build build
~~~

### 3. Run correctness tests

All registered suites:

~~~bat
ctest --test-dir build --output-on-failure
~~~

Alternatively, run individual suites:

~~~bat
build\test_matmul.exe
build\test_layernorm.exe
build\test_softmax.exe
build\test_edge_cases.exe
python -X utf8 tests\test_analysis.py
~~~

### 4. Full target benchmark and final comparison table

This regenerates the main benchmark dataset using five trials and fifty samples per trial.

~~~bat
build\benchmark_targets.exe --trials 5 --samples 50 --csv results\target_results.csv
python -X utf8 scripts\analyze_results.py --csv results\target_results.csv --plot
~~~

To display the saved comparison again without running the GPU benchmarks:

~~~bat
python -X utf8 scripts\analyze_results.py --csv results\target_results.csv
~~~

The CSV, individual trial CSV, metadata, summary and optional chart are saved under results.

### 5. Short live demo: LayerNorm versus cuDNN

~~~bat
ctest --test-dir build --output-on-failure
build\benchmark_targets.exe --layernorm 128 768 --trials 5 --samples 50 --csv results\demo.csv
python -X utf8 scripts\analyze_results.py --csv results\demo.csv --out-dir results\demo
python -X utf8 scripts\analyze_results.py --csv results\target_results.csv
~~~

The first analysis displays the live measurement. The last command displays the full saved evaluation; identify it as a saved run during the presentation.

### 6. Three-operator presentation demo

~~~bat
build\benchmark_targets.exe --matmul 128 64 128 --layernorm 128 768 --softmax 12288 1024 --trials 5 --samples 50 --csv results\demo.csv
python -X utf8 scripts\analyze_results.py --csv results\demo.csv --out-dir results\demo --plot
~~~

The demo CSV and output directory keep the main target_results dataset separate.

### 7. Individual operator benchmarks with vendor comparisons

MatMul arguments are M K N:

~~~bat
build\benchmark_targets.exe --matmul 128 64 128 --trials 5 --samples 50 --csv results\matmul_demo.csv
python -X utf8 scripts\analyze_results.py --csv results\matmul_demo.csv --out-dir results\matmul_demo
~~~

LayerNorm arguments are rows hidden:

~~~bat
build\benchmark_targets.exe --layernorm 128 768 --trials 5 --samples 50 --csv results\layernorm_demo.csv
python -X utf8 scripts\analyze_results.py --csv results\layernorm_demo.csv --out-dir results\layernorm_demo
~~~

Softmax arguments are rows columns:

~~~bat
build\benchmark_targets.exe --softmax 12288 1024 --trials 5 --samples 50 --csv results\softmax_demo.csv
python -X utf8 scripts\analyze_results.py --csv results\softmax_demo.csv --out-dir results\softmax_demo
~~~

### 8. Quick smoke check

~~~bat
build\benchmark_targets.exe --quick --trials 3 --samples 20 --csv results\smoke.csv
python -X utf8 scripts\analyze_results.py --csv results\smoke.csv --out-dir results\smoke
~~~

### 9. Legacy and standalone executables

These print variant timings. Use benchmark_targets for the principal documented vendor-comparison workflow.

~~~bat
build\benchmark_matmul.exe 1024 1024 1024
build\benchmark_layernorm.exe 128 4096
build\benchmark_softmax.exe 128 4096
build\cuda_kernel_library.exe
~~~

Running the legacy full-suite executable from the project root writes results/benchmark_results.csv:

~~~bat
python -X utf8 scripts\analyze_results.py --csv results\benchmark_results.csv --out-dir results\legacy
~~~

### 10. GPU memory and shared-memory race checks

~~~bat
compute-sanitizer --tool memcheck --error-exitcode 99 --log-file results\memcheck.log build\test_edge_cases.exe
compute-sanitizer --tool racecheck --error-exitcode 99 --log-file results\racecheck.log build\test_edge_cases.exe
~~~

These checks run separately from ordinary performance measurements.

### 11. Nsight Compute profiling

Profile one operator:

~~~bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\profile.ps1 -Operator MatMul
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\profile.ps1 -Operator LayerNorm
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\profile.ps1 -Operator Softmax
~~~

Or collect all three reports:

~~~bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\profile.ps1
~~~

Open a recorded report in the GUI or read its details in the terminal:

~~~bat
ncu-ui results\profiles\layernorm.ncu-repz
ncu --import results\profiles\layernorm.ncu-repz --page details
~~~

Profile-instrumented timings remain under results/profiles and must not be used as the normal benchmark speedup evidence.

### 12. Help and automated workflow

~~~bat
build\benchmark_targets.exe --help
python scripts\analyze_results.py --help
~~~

The automated build, test, benchmark and analysis workflow is:

~~~bat
run_all.bat
~~~

### 13. Custom dimensions

This example runs MatMul `[256x512] * [512x256]`, LayerNorm with 256 rows of 512 features, and Softmax with 256 rows of 512 scores. No rebuild is needed for a dimension change.

~~~bat
build\benchmark_targets.exe --matmul 256 512 256 --layernorm 256 512 --softmax 256 512 --trials 5 --samples 50 --csv results\custom_results.csv
python -X utf8 scripts\analyze_results.py --csv results\custom_results.csv --out-dir results\custom
~~~

The benchmark also writes `results\custom_results.trials.csv` and `results\custom_results.metadata.json`. Analysis writes `results\custom\summary.md`. Add `--plot` to generate `results\custom\benchmark_chart.png`.

To test only one new MatMul shape:

~~~bat
build\benchmark_targets.exe --matmul 64 256 128 --trials 5 --samples 50 --csv results\custom_matmul.csv
python -X utf8 scripts\analyze_results.py --csv results\custom_matmul.csv --out-dir results\custom_matmul
~~~

### 14. Command-line options

| Program / option | Meaning |
|---|---|
| `benchmark_targets --matmul M K N` | Add a MatMul case; repeat for multiple shapes |
| `benchmark_targets --layernorm ROWS HIDDEN` | Add a LayerNorm case |
| `benchmark_targets --softmax ROWS COLS` | Add a Softmax case |
| `benchmark_targets --trials N` | Independent timing trials; default 5; at least 3 needed for confirmed-win evidence |
| `benchmark_targets --samples N` | Timing samples per trial; default 50 |
| `benchmark_targets --quick` | Reduced default workload: two MatMul shapes, one LayerNorm, and one Softmax; does not change trial/sample counts |
| `benchmark_targets --csv PATH` | Aggregate CSV destination; also determines trial and metadata filenames |
| `analyze_results.py --csv PATH` | Analyze this saved CSV; no GPU benchmark execution |
| `analyze_results.py --out-dir PATH` | Destination for summary and optional chart; default `results` under the project root |
| `analyze_results.py --metric med_ms` | Compare aggregate median timings; the default |
| `analyze_results.py --metric avg_ms` | Compare aggregate averages; does not certify median-based confirmed wins |
| `analyze_results.py --plot` | Generate a chart if matplotlib is available |

In the option table, `benchmark_targets` refers to `build\benchmark_targets.exe`. Relative CSV paths are resolved from the terminal's working directory. Default output filenames are reused on subsequent runs; choose a distinct CSV and analysis output directory to preserve an experiment. An explicit shape selection takes precedence over the reduced/default shape sets.
