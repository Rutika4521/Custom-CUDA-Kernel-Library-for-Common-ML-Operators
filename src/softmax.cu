#include "validation.cuh"
// =============================================================================
// softmax.cu — Softmax Kernel Implementations
// Phase 8: V1→V4
// =============================================================================

#include "softmax.cuh"
#include "cuda_utils.cuh"
#include <cmath>
#include <cfloat>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3 — CPU Reference: numerically stable softmax
// For each row: max → shift → exp → sum → normalize
// ─────────────────────────────────────────────────────────────────────────────
void cpu_softmax(const float* x, float* y, int batch, int cols)
{
    for (int b = 0; b < batch; ++b) {
        const float* xb = x + b * cols;
        float*       yb = y + b * cols;

        float mx = xb[0];
        for (int i = 1; i < cols; ++i) mx = fmaxf(mx, xb[i]);

        double sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            yb[i] = expf(xb[i] - mx);
            sum += yb[i];
        }
        for (int i = 0; i < cols; ++i) yb[i] /= (float)sum;
    }
}

// Helper: intra-warp max
__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// Helper: intra-warp sum
__device__ inline float warp_reduce_sum_s(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// =============================================================================
// V1: Shared-memory, three passes.
// Pass 1: find max (parallel reduce in shmem)
// Pass 2: compute exp, sum (parallel reduce in shmem)
// Pass 3: normalize
// =============================================================================
__global__
void kernel_softmax_v1(const float* __restrict__ x,
                        float*       __restrict__ y,
                        int cols)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* xrow = x + row * cols;
    float*       yrow = y + row * cols;

    // ── Pass 1: max ───────────────────────────────────────────────────────
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, xrow[i]);
    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float mx = smem[0];
    __syncthreads();

    // ── Pass 2: exp + sum ─────────────────────────────────────────────────
    float local_sum = 0.f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float ev = expf(xrow[i] - mx);
        yrow[i]  = ev;          // store exp temporarily
        local_sum += ev;
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.f / smem[0];
    __syncthreads();

    // ── Pass 3: normalize ─────────────────────────────────────────────────
    for (int i = tid; i < cols; i += blockDim.x)
        yrow[i] *= inv_sum;
}

void launch_softmax_v1(const float* d_x, float* d_y, int batch, int cols)
{
    validate_softmax(d_x,d_y,batch,cols); if (!batch) return;

    int threads = 256;
    size_t smem = threads * sizeof(float);
    kernel_softmax_v1<<<batch, threads, smem, execution_stream()>>>(d_x, d_y, cols);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// V2: Warp-level reduction for max; shared-mem for cross-warp max.
// =============================================================================
__global__
void kernel_softmax_v2(const float* __restrict__ x,
                        float*       __restrict__ y,
                        int cols)
{
    extern __shared__ float smem[];
    int tid       = threadIdx.x;
    int wid       = tid / 32;
    int lane      = tid % 32;
    int num_warps = blockDim.x / 32;
    int row       = blockIdx.x;
    const float* xrow = x + row * cols;
    float*       yrow = y + row * cols;

    // ── Warp-level max ────────────────────────────────────────────────────
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, xrow[i]);
    local_max = warp_reduce_max(local_max);
    if (lane == 0) smem[wid] = local_max;
    __syncthreads();

    float gmax = (tid < num_warps) ? smem[tid] : -FLT_MAX;
    if (wid == 0) {
        gmax = warp_reduce_max(gmax);
        if (tid == 0) smem[0] = gmax;
    }
    __syncthreads();
    float mx = smem[0];
    __syncthreads();

    // ── Exp + shared-mem sum ──────────────────────────────────────────────
    float local_sum = 0.f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float ev = expf(xrow[i] - mx);
        yrow[i]  = ev;
        local_sum += ev;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.f / smem[0];
    __syncthreads();

    for (int i = tid; i < cols; i += blockDim.x)
        yrow[i] *= inv_sum;
}

void launch_softmax_v2(const float* d_x, float* d_y, int batch, int cols)
{
    validate_softmax(d_x,d_y,batch,cols); if (!batch) return;

    int threads   = 256;
    int num_warps = threads / 32;
    size_t smem   = fmaxf(threads, num_warps) * sizeof(float);
    kernel_softmax_v2<<<batch, threads, smem, execution_stream()>>>(d_x, d_y, cols);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// V3: Warp-level reductions for both max and sum.
// =============================================================================
__global__
void kernel_softmax_v3(const float* __restrict__ x,
                        float*       __restrict__ y,
                        int cols)
{
    extern __shared__ float smem[];
    int tid       = threadIdx.x;
    int wid       = tid / 32;
    int lane      = tid % 32;
    int num_warps = blockDim.x / 32;
    int row       = blockIdx.x;
    const float* xrow = x + row * cols;
    float*       yrow = y + row * cols;

    auto block_max = [&](float lv) -> float {
        lv = warp_reduce_max(lv);
        if (lane == 0) smem[wid] = lv;
        __syncthreads();
        lv = (tid < num_warps) ? smem[tid] : -FLT_MAX;
        if (wid == 0) {
            lv = warp_reduce_max(lv);
            if (tid == 0) smem[0] = lv;
        }
        __syncthreads();
        return smem[0];
    };

    auto block_sum = [&](float lv) -> float {
        lv = warp_reduce_sum_s(lv);
        if (lane == 0) smem[wid] = lv;
        __syncthreads();
        lv = (tid < num_warps) ? smem[tid] : 0.f;
        if (wid == 0) {
            lv = warp_reduce_sum_s(lv);
            if (tid == 0) smem[0] = lv;
        }
        __syncthreads();
        return smem[0];
    };

    // Max
    float lmax = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
        lmax = fmaxf(lmax, xrow[i]);
    float mx = block_max(lmax);
    __syncthreads();

    // Exp + sum
    float lsum = 0.f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float ev = expf(xrow[i] - mx);
        yrow[i]  = ev;
        lsum    += ev;
    }
    float total   = block_sum(lsum);
    float inv_sum = 1.f / total;

    for (int i = tid; i < cols; i += blockDim.x)
        yrow[i] *= inv_sum;
}

void launch_softmax_v3(const float* d_x, float* d_y, int batch, int cols)
{
    validate_softmax(d_x,d_y,batch,cols); if (!batch) return;

    int threads   = 256;
    int num_warps = threads / 32;
    size_t smem   = num_warps * sizeof(float);
    kernel_softmax_v3<<<batch, threads, smem, execution_stream()>>>(d_x, d_y, cols);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// V4: Online (single-pass) softmax — Milakov & Gimelshein 2018 algorithm.
// Computes max and sum simultaneously, avoiding two passes over global memory.
//
// Algorithm (per thread):
//   For each element xk:
//     if xk > old_max:
//       sum = sum * exp(old_max - xk) + exp(0)   // rescale old sum
//       max = xk
//     else:
//       sum += exp(xk - max)
//
// After warp/block reduction, output: exp(x_i - max) / sum
//
// Note: still needs one pass to write output after computing normalisation.
// =============================================================================
struct MaxSum { float max_val; float sum; };

__device__ inline MaxSum merge_maxsum(MaxSum a, MaxSum b) {
    if (a.max_val > b.max_val)
        return { a.max_val, a.sum + b.sum * expf(b.max_val - a.max_val) };
    else
        return { b.max_val, a.sum * expf(a.max_val - b.max_val) + b.sum };
}

__device__ inline MaxSum warp_reduce_maxsum(MaxSum ms) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float rm = __shfl_down_sync(0xFFFFFFFF, ms.max_val, offset);
        float rs = __shfl_down_sync(0xFFFFFFFF, ms.sum,     offset);
        ms = merge_maxsum(ms, {rm, rs});
    }
    return ms;
}

__global__
void kernel_softmax_v4(const float* __restrict__ x,
                        float*       __restrict__ y,
                        int cols)
{
    // smem: [2 * num_warps] for max_val and sum from each warp
    extern __shared__ float smem[];
    int tid       = threadIdx.x;
    int wid       = tid / 32;
    int lane      = tid % 32;
    int num_warps = blockDim.x / 32;
    int row       = blockIdx.x;

    const float* xrow = x + row * cols;
    float*       yrow = y + row * cols;

    // ── Online max+sum accumulation with float4 vectorized reads ─────────
    MaxSum local_ms = { -FLT_MAX, 0.f };
    const float4* xrow4 = reinterpret_cast<const float4*>(xrow);
    int vec4_end = (cols / 4) * 4;

    for (int i4 = tid; i4 < cols / 4; i4 += blockDim.x) {
        float4 v = xrow4[i4];
        float vals[4] = { v.x, v.y, v.z, v.w };
        for (int j = 0; j < 4; ++j) {
            MaxSum elem = { vals[j], 1.f };
            local_ms = merge_maxsum(local_ms, elem);
        }
    }
    for (int i = vec4_end + tid; i < cols; i += blockDim.x) {
        local_ms = merge_maxsum(local_ms, {xrow[i], 1.f});
    }

    // ── Warp-level reduction ──────────────────────────────────────────────
    local_ms = warp_reduce_maxsum(local_ms);
    float* smem_max = smem;
    float* smem_sum = smem + num_warps;

    if (lane == 0) {
        smem_max[wid] = local_ms.max_val;
        smem_sum[wid] = local_ms.sum;
    }
    __syncthreads();

    // ── Cross-warp reduction (warp 0) ─────────────────────────────────────
    if (wid == 0) {
        MaxSum gms = {
            (tid < num_warps) ? smem_max[tid] : -FLT_MAX,
            (tid < num_warps) ? smem_sum[tid] :  0.f
        };
        gms = warp_reduce_maxsum(gms);
        if (tid == 0) {
            smem_max[0] = gms.max_val;
            smem_sum[0] = gms.sum;
        }
    }
    __syncthreads();

    float global_max = smem_max[0];
    float global_sum = smem_sum[0];
    float inv_sum    = 1.f / global_sum;

    // ── Write normalized output with vectorized writes ─────────────────────
    float4* yrow4 = reinterpret_cast<float4*>(yrow);
    for (int i4 = tid; i4 < cols / 4; i4 += blockDim.x) {
        float4 v = xrow4[i4];
        yrow4[i4] = {
            expf(v.x - global_max) * inv_sum,
            expf(v.y - global_max) * inv_sum,
            expf(v.z - global_max) * inv_sum,
            expf(v.w - global_max) * inv_sum
        };
    }
    for (int i = vec4_end + tid; i < cols; i += blockDim.x)
        yrow[i] = expf(xrow[i] - global_max) * inv_sum;
}

void launch_softmax_v4(const float* d_x, float* d_y, int batch, int cols)
{
    validate_softmax(d_x,d_y,batch,cols); if (!batch) return;
    if (cols%4 || !aligned16(d_x) || !aligned16(d_y)) { launch_softmax_v3(d_x,d_y,batch,cols); return; }

    int threads   = 256;
    int num_warps = threads / 32;
    size_t smem   = 2 * num_warps * sizeof(float);
    kernel_softmax_v4<<<batch, threads, smem, execution_stream()>>>(d_x, d_y, cols);
    CUDA_CHECK(cudaGetLastError());
}
