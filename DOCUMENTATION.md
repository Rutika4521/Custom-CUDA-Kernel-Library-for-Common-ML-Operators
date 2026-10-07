# Implementation and experimental methodology

## Objective and scope

Investigate hand-tuned FP32 CUDA implementations of MatMul, LayerNorm and Softmax for documented model-related shapes. Completion means reproducible builds, validated outputs, comparable vendor measurements and profiling evidence. A speedup claim is limited to the actual shape, GPU, precision, library version and timing protocol.

The reference workload is specified in TARGET_WORKLOAD.md. Synthetic inputs are deterministic; this repository does not implement an end-to-end GPT-2 model or measure production traffic.

## Mathematics

MatMul: C[M,N] = A[M,K] B[K,N], with C[i,j] = sum_k A[i,k] B[k,j]. The conventional operation count is 2*M*K*N. Inputs and outputs are contiguous row-major FP32. cuBLAS receives swapped operands to compute C-transpose = B-transpose A-transpose without moving the data.

LayerNorm computes statistics independently across each row: mean = sum(x)/H; variance = sum((x-mean)^2)/H; y[i] = gamma[i]*(x[i]-mean)/sqrt(variance+epsilon)+beta[i]. Gamma and beta are feature vectors. Default epsilon is 1e-5. The variance divisor is H, not H-1.

Softmax: m=max(x); s=sum(exp(x-m)); y=exp(x-m)/s. Subtracting the maximum preserves the normalized result and prevents positive exponential overflow for finite inputs. Output rows are nonnegative and sum to approximately 1.

## Implementations

MatMul V1 assigns one output element to each thread. V2 cooperatively loads 16x16 or 32x32 tiles into shared memory, synchronizes, computes and synchronizes before reusing storage. V3 computes a 4x4 output patch per thread using local accumulators. V4 uses scalar shared-memory tile loads and grouped/unrolled multiply-adds; it has no explicit float4 global loads.

The shape kernel uses coalesced cooperative tile loads, transposed/padded A storage and interleaved output columns to reduce shared-memory bank conflicts. Ordinary shapes use a 64x64 output tile, K depth 16, 256 threads and 4x4 register patches. Short row counts use 16x64 output tiles. A single input row uses a GEMV specialization that splits K across eight warps and combines their partial sums.

LayerNorm V1 uses shared-memory tree reductions. V2/V3 use warp shuffles plus shared memory and barriers between warps. V4 accumulates Welford statistics locally, merges tuples of mean/M2/count, then rereads input to normalize. Its float4 path requires both alignment and compatible row stride; otherwise it falls back to V3.

The resident LayerNorm kernel retains a row's values in registers through both statistics reductions and output. Transformer widths use 128 threads per row with 6/8/24/32 values per thread, depending on width. Small widths use one warp per row. This reduces input traffic while avoiding the high register pressure of a wide row assigned entirely to one warp.

Softmax V1 stores intermediate exponentials in output memory and performs shared-memory max/sum reductions. V2 moves the maximum reduction into warps. V3 uses warp reductions for both statistics with shared-memory cross-warp communication. V4 uses online max/sum merging and float4 reads, with a safe fallback for alignment/stride.

The shape Softmax assigns one warp to a row and four rows to a block. Values remain in registers through max, exponential sum and output, removing intermediate global-memory traffic. It uses the CUDA __expf intrinsic explicitly; all implementations are validated numerically before timing. Widths above 4096 use V3.

## CUDA execution and API contract

A grid contains blocks, blocks contain threads, and threads are scheduled in warps of 32. Blocks cooperate through shared memory and __syncthreads. Warp shuffles exchange register values inside a warp; they do not replace synchronization between warps.

Public launchers validate host arguments before launching. Rows may be zero (no-op); feature/reduction dimensions must be positive. Null buffers for nonempty work, invalid epsilon and element counts above INT_MAX are rejected. Buffers must be contiguous FP32 device memory and respect the non-aliasing contract. General finite ranges and adversarial data require their own numerical validation; passing the provided tests is not a proof for all possible inputs.

Operations use stream 0 by default. cuda_utils.cuh exposes a thread-local execution stream used internally for benchmark capture. If applications select a nondefault stream, they must synchronize it before reading results with their own transfer code. cuBLAS handles should not be shared concurrently between threads without appropriate coordination.

## Dispatcher and tuning

The dispatcher checks a mutex-protected process-local cache keyed by operator, shape and CUDA device. MatMul keys distinguish availability of a vendor handle. LayerNorm keys encode epsilon's exact FP32 bits.

The tuner initializes deterministic finite inputs, computes an independent reference (CPU for norms, cuBLAS for MatMul), rejects incorrect candidates and compares medians across three timing trials. MatMul includes V1-V4, the shape kernel, both V2 tile sizes and cuBLAS when a handle is supplied. Norm tuners compare the four legacy variants and the shape kernel. The target benchmark evaluates the tuned custom dispatcher alongside individual variants. Tuning cost is excluded from steady-state measurements.

## Benchmark protocol

The target benchmark uses FP32 inputs/outputs. cuBLAS runs SGEMM in pedantic FP32 math mode. cuDNN LayerNorm uses its inference graph and a plan chosen by vendor heuristics; cuDNN Softmax uses accurate instance mode. Plan creation and buffer binding occur outside replayed GPU timing. These are versioned default vendor baselines, not an exhaustive search over all vendor configurations or precisions.

All outputs are checked before measurement. NaN/Inf is rejected. The target comparison uses per-element absolute plus relative tolerance (1e-4 + 1e-4*abs(reference)); original standalone correctness suites use their specified absolute tolerances. Softmax additionally checks nonnegative values and row sums within 1e-4.

A benchmark performs 20 warm-ups, calibrates a repeat count capped at 64, captures repeated launches on a dedicated CUDA stream, and times graph replay with CUDA events. Each event interval is divided by the captured launch count. Default evaluation uses 5 independent trials and 50 samples per trial; variant order rotates between trials. Allocation, host-device transfers, tuning, graph capture/instantiation and plan setup are excluded. Reused buffers can be cache-resident. Results measure steady-state operator execution, not model latency or CPU launch latency.

Aggregate average = mean of trial averages; aggregate median = median of trial medians. The trials CSV retains individual measurements. Metadata records GPU, runtime/driver API versions, precision, seeds and timing settings. TFLOPS = FLOPs/(average_ms*1e9). Norm TFLOPS is not supplied; zero in that column does not mean zero work.

Speedup against V1 = V1 latency/custom latency. Vendor speedup = vendor latency/custom latency. Ratios above 1 indicate improvement. A reported repeatable vendor win requires at least three trials and a ratio >=1.05 in every trial, evaluated using medians. This is a conservative repeatability criterion, not a statistical confidence interval. Losses remain in the report. Hardware clock/power behavior and workload ordering can still affect results.

## Profiling

scripts/profile.ps1 profiles one representative shape for each operator with Nsight Compute's full metric set. Reports are stored under results/profiles. Profiler-instrumented CSVs are kept separate from benchmark evidence. Basic/full report inspection can reveal register usage, occupancy, memory throughput, shared-memory conflicts and stalls; low throughput alone does not establish a single bottleneck.

The initial LayerNorm profile found 80 registers/thread, 50% theoretical occupancy, 13.31% achieved occupancy and only 0.27 waves/SM. This motivated a 128-thread resident-row kernel with fewer retained values per thread. The initial report is layernorm_initial.ncu-repz; final reports describe the revised implementation. Profiling duration must not be compared directly to uninstrumented graph timings.

Compute Sanitizer memcheck validates memory accesses; racecheck checks shared-memory hazards. Logs accompany the final validation. Test coverage includes odd row widths, offset device pointers, constants, extreme finite Softmax logits, dispatcher selections and rejection of nonfinite comparisons.

## Reproduction and limitations

Use build.bat, CTest and benchmark_targets as documented in README.md. run_all.bat propagates failures instead of continuing with misleading success. Analyzer paths are independent of the working directory when auto-detected, and --csv can explicitly select a dataset. Optional charts group comparable shapes and separate operators.

The project currently implements forward FP32 operators on one GPU. Unsupported extensions include backward/training, arbitrary tensor strides, batched GEMM, FP16/BF16/INT8, Tensor Core kernels, causal masked/fused attention, persistent tuning files and PyTorch integration. There is no claim of vendor superiority across all shapes or GPUs. Read the recorded results for the actual scope of any measured speedup.
