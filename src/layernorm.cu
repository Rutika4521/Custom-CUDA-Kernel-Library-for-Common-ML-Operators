#include "validation.cuh"
// =============================================================================
// layernorm.cu — Layer Normalization Kernel Implementations
// Phase 7: V1→V4
// =============================================================================

#include "layernorm.cuh"
#include "cuda_utils.cuh"
#include <cmath>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3 — CPU Reference LayerNorm
// mean = sum(x)/N
// var  = sum((x-mean)^2)/N
// y_i  = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i
// ─────────────────────────────────────────────────────────────────────────────
void cpu_layernorm(const float* x, const float* gamma, const float* beta,
                   float* y, int batch, int hidden, float eps)
{
    for (int b = 0; b < batch; ++b) {
        const float* xb = x + b * hidden;
        float*       yb = y + b * hidden;

        // mean
        double mean = 0.0;
        for (int i = 0; i < hidden; ++i) mean += xb[i];
        mean /= hidden;

        // variance
        double var = 0.0;
        for (int i = 0; i < hidden; ++i) {
            double d = xb[i] - mean;
            var += d * d;
        }
        var /= hidden;

        float inv_std = 1.f / sqrtf((float)var + eps);
        for (int i = 0; i < hidden; ++i)
            yb[i] = gamma[i] * ((xb[i] - (float)mean) * inv_std) + beta[i];
    }
}

// =============================================================================
// V1: Basic shared-memory LayerNorm
// One block handles one row.
// Pass 1: parallel sum → mean
// Pass 2: parallel sum of squared deviation → variance
// Pass 3: normalize
// =============================================================================
__global__
void kernel_layernorm_v1(const float* __restrict__ x,
                          const float* __restrict__ gamma,
                          const float* __restrict__ beta,
                          float* __restrict__ y,
                          int hidden, float eps)
{
    extern __shared__ float smem[];  // [blockDim.x] floats
    int tid  = threadIdx.x;
    int row  = blockIdx.x;
    const float* xrow = x + row * hidden;
    float*       yrow = y + row * hidden;

    // ── Pass 1: partial sum ───────────────────────────────────────────────
    float local_sum = 0.f;
    for (int i = tid; i < hidden; i += blockDim.x)
        local_sum += xrow[i];
    smem[tid] = local_sum;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / hidden;
    __syncthreads();

    // ── Pass 2: variance ─────────────────────────────────────────────────
    float local_var = 0.f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float d = xrow[i] - mean;
        local_var += d * d;
    }
    smem[tid] = local_var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / hidden + eps);
    __syncthreads();

    // ── Pass 3: normalize ─────────────────────────────────────────────────
    for (int i = tid; i < hidden; i += blockDim.x)
        yrow[i] = gamma[i] * ((xrow[i] - mean) * inv_std) + beta[i];
}

void launch_layernorm_v1(const float* d_x, const float* d_gamma,
                          const float* d_beta, float* d_y,
                          int batch, int hidden, float eps)
{
    validate_layernorm(d_x,d_gamma,d_beta,d_y,batch,hidden,eps); if (!batch) return;

    int threads = 256;
    size_t smem = threads * sizeof(float);
    kernel_layernorm_v1<<<batch, threads, smem, execution_stream()>>>(d_x, d_gamma, d_beta, d_y, hidden, eps);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// V2: Efficient shared-memory reduction (warp-aligned)
// Uses explicit power-of-2 reduction within shared memory.
// =============================================================================
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__
void kernel_layernorm_v2(const float* __restrict__ x,
                          const float* __restrict__ gamma,
                          const float* __restrict__ beta,
                          float* __restrict__ y,
                          int hidden, float eps)
{
    // blockDim.x = warp_size * num_warps
    extern __shared__ float smem[];  // [num_warps]
    int tid      = threadIdx.x;
    int wid      = tid / 32;
    int lane     = tid % 32;
    int num_warps = blockDim.x / 32;
    int row      = blockIdx.x;

    const float* xrow = x + row * hidden;
    float*       yrow = y + row * hidden;

    // ── Mean: per-warp reduction then cross-warp ──────────────────────────
    float local_sum = 0.f;
    for (int i = tid; i < hidden; i += blockDim.x)
        local_sum += xrow[i];
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) smem[wid] = local_sum;
    __syncthreads();

    float v_mean = (tid < num_warps) ? smem[tid] : 0.f;
    if (wid == 0) {
        v_mean = warp_reduce_sum(v_mean);
        if (tid == 0) smem[0] = v_mean / hidden;
    }
    __syncthreads();
    float mean = smem[0];
    __syncthreads();

    // ── Variance ─────────────────────────────────────────────────────────
    float local_var = 0.f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float d = xrow[i] - mean;
        local_var += d * d;
    }
    local_var = warp_reduce_sum(local_var);
    if (lane == 0) smem[wid] = local_var;
    __syncthreads();

    float v_var = (tid < num_warps) ? smem[tid] : 0.f;
    if (wid == 0) {
        v_var = warp_reduce_sum(v_var);
        if (tid == 0) smem[0] = rsqrtf(v_var / hidden + eps);
    }
    __syncthreads();
    float inv_std = smem[0];

    // ── Normalize ─────────────────────────────────────────────────────────
    for (int i = tid; i < hidden; i += blockDim.x)
        yrow[i] = gamma[i] * ((xrow[i] - mean) * inv_std) + beta[i];
}

void launch_layernorm_v2(const float* d_x, const float* d_gamma,
                          const float* d_beta, float* d_y,
                          int batch, int hidden, float eps)
{
    validate_layernorm(d_x,d_gamma,d_beta,d_y,batch,hidden,eps); if (!batch) return;

    int threads   = 256;
    int num_warps = threads / 32;
    size_t smem   = num_warps * sizeof(float);
    kernel_layernorm_v2<<<batch, threads, smem, execution_stream()>>>(d_x, d_gamma, d_beta, d_y, hidden, eps);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// V3: Warp-level primitives only — __shfl_down_sync for both reductions.
// No shared-memory read-after-write latency within warp.
// =============================================================================
__global__
void kernel_layernorm_v3(const float* __restrict__ x,
                          const float* __restrict__ gamma,
                          const float* __restrict__ beta,
                          float* __restrict__ y,
                          int hidden, float eps)
{
    extern __shared__ float smem[];   // [num_warps] for cross-warp reduction
    int tid       = threadIdx.x;
    int wid       = tid / 32;
    int lane      = tid % 32;
    int num_warps = blockDim.x / 32;
    int row       = blockIdx.x;

    const float* xrow = x + row * hidden;
    float*       yrow = y + row * hidden;

    // Helper: intra-warp + cross-warp reduction
    auto block_reduce = [&](float local_val) -> float {
        // intra-warp
        local_val = warp_reduce_sum(local_val);
        if (lane == 0) smem[wid] = local_val;
        __syncthreads();
        // cross-warp (warp 0)
        local_val = (tid < num_warps) ? smem[tid] : 0.f;
        if (wid == 0) {
            local_val = warp_reduce_sum(local_val);
            if (tid == 0) smem[0] = local_val;
        }
        __syncthreads();
        return smem[0];
    };

    // ── Mean ─────────────────────────────────────────────────────────────
    float s = 0.f;
    for (int i = tid; i < hidden; i += blockDim.x) s += xrow[i];
    float mean = block_reduce(s) / hidden;
    __syncthreads();

    // ── Variance ──────────────────────────────────────────────────────────
    float v = 0.f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float d = xrow[i] - mean; v += d * d;
    }
    float inv_std = rsqrtf(block_reduce(v) / hidden + eps);

    // ── Normalize ─────────────────────────────────────────────────────────
    for (int i = tid; i < hidden; i += blockDim.x)
        yrow[i] = gamma[i] * ((xrow[i] - mean) * inv_std) + beta[i];
}

void launch_layernorm_v3(const float* d_x, const float* d_gamma,
                          const float* d_beta, float* d_y,
                          int batch, int hidden, float eps)
{
    validate_layernorm(d_x,d_gamma,d_beta,d_y,batch,hidden,eps); if (!batch) return;

    int threads   = 256;
    int num_warps = threads / 32;
    size_t smem   = num_warps * sizeof(float);
    kernel_layernorm_v3<<<batch, threads, smem, execution_stream()>>>(d_x, d_gamma, d_beta, d_y, hidden, eps);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// V4: Vectorized loads + single-pass Welford online mean/variance.
// Welford's algorithm computes mean and variance in a single pass:
//   For each element x:
//     n  += 1
//     delta = x - mean
//     mean += delta / n
//     M2   += delta * (x - mean)  (M2 = sum of squared deviations)
// This eliminates the second global-memory pass over x.
// =============================================================================
__global__
void kernel_layernorm_v4(const float* __restrict__ x,
                          const float* __restrict__ gamma,
                          const float* __restrict__ beta,
                          float* __restrict__ y,
                          int hidden, float eps)
{
    extern __shared__ float smem[];   // [2 * num_warps]
    int tid       = threadIdx.x;
    int wid       = tid / 32;
    int lane      = tid % 32;
    int num_warps = blockDim.x / 32;
    int row       = blockIdx.x;

    const float* xrow = x + row * hidden;
    float*       yrow = y + row * hidden;

    // ── Single-pass Welford accumulation (per-thread) ─────────────────────
    float local_mean = 0.f, local_M2 = 0.f;
    int   local_n    = 0;

    // Use float4 for vectorized 128-bit loads where possible
    int vec4_end = (hidden / 4) * 4;
    const float4* xrow4 = reinterpret_cast<const float4*>(xrow);

    for (int i4 = tid; i4 < hidden / 4; i4 += blockDim.x) {
        float4 v = xrow4[i4];
        float vals[4] = { v.x, v.y, v.z, v.w };
        for (int j = 0; j < 4; ++j) {
            float xv = vals[j];
            local_n++;
            float delta = xv - local_mean;
            local_mean += delta / local_n;
            local_M2   += delta * (xv - local_mean);
        }
    }
    // Handle tail elements
    for (int i = vec4_end + tid; i < hidden; i += blockDim.x) {
        float xv = xrow[i]; local_n++;
        float delta = xv - local_mean;
        local_mean += delta / local_n;
        local_M2   += delta * (xv - local_mean);
    }

    // ── Parallel Welford merge across threads ─────────────────────────────
    // Intra-warp merge using shfl
    for (int offset = 16; offset > 0; offset >>= 1) {
        float rm = __shfl_down_sync(0xFFFFFFFF, local_mean, offset);
        float rM = __shfl_down_sync(0xFFFFFFFF, local_M2,   offset);
        int   rn = __shfl_down_sync(0xFFFFFFFF, local_n,    offset);
        if (local_n + rn > 0) {
            float delta = rm - local_mean;
            int   total = local_n + rn;
            local_mean  = (local_n * local_mean + rn * rm) / total;
            local_M2   += rM + delta * delta * local_n * rn / total;
            local_n     = total;
        }
    }

    // Cross-warp: warp lane 0 writes to smem
    float* smem_mean = smem;
    float* smem_M2   = smem + num_warps;
    int*   smem_n    = (int*)(smem + 2 * num_warps);

    if (lane == 0) {
        smem_mean[wid] = local_mean;
        smem_M2[wid]   = local_M2;
        smem_n[wid]    = local_n;
    }
    __syncthreads();

    // Warp 0 reduces
    if (wid == 0) {
        float gm = (tid < num_warps) ? smem_mean[tid] : 0.f;
        float gM = (tid < num_warps) ? smem_M2[tid]   : 0.f;
        int   gn = (tid < num_warps) ? smem_n[tid]     : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float rm = __shfl_down_sync(0xFFFFFFFF, gm, offset);
            float rM = __shfl_down_sync(0xFFFFFFFF, gM, offset);
            int   rn = __shfl_down_sync(0xFFFFFFFF, gn, offset);
            if (gn + rn > 0) {
                float delta = rm - gm;
                int   total = gn + rn;
                gm  = (gn * gm + rn * rm) / total;
                gM += rM + delta * delta * gn * rn / total;
                gn  = total;
            }
        }
        if (tid == 0) {
            smem_mean[0] = gm;
            smem_M2[0]   = gM;
        }
    }
    __syncthreads();

    float mean    = smem_mean[0];
    float inv_std = rsqrtf(smem_M2[0] / hidden + eps);

    // ── Vectorized write ──────────────────────────────────────────────────
    float4* yrow4       = reinterpret_cast<float4*>(yrow);
    const float4* g4    = reinterpret_cast<const float4*>(gamma);
    const float4* b4    = reinterpret_cast<const float4*>(beta);

    for (int i4 = tid; i4 < hidden / 4; i4 += blockDim.x) {
        float4 xv = xrow4[i4];
        float4 gv = g4[i4];
        float4 bv = b4[i4];
        yrow4[i4] = {
            gv.x * ((xv.x - mean) * inv_std) + bv.x,
            gv.y * ((xv.y - mean) * inv_std) + bv.y,
            gv.z * ((xv.z - mean) * inv_std) + bv.z,
            gv.w * ((xv.w - mean) * inv_std) + bv.w
        };
    }
    for (int i = vec4_end + tid; i < hidden; i += blockDim.x)
        yrow[i] = gamma[i] * ((xrow[i] - mean) * inv_std) + beta[i];
}

void launch_layernorm_v4(const float* d_x, const float* d_gamma,
                          const float* d_beta, float* d_y,
                          int batch, int hidden, float eps)
{
    validate_layernorm(d_x,d_gamma,d_beta,d_y,batch,hidden,eps); if (!batch) return;
    if (hidden%4 || !aligned16(d_x) || !aligned16(d_gamma) || !aligned16(d_beta) || !aligned16(d_y)) { launch_layernorm_v3(d_x,d_gamma,d_beta,d_y,batch,hidden,eps); return; }

    int threads   = 256;
    int num_warps = threads / 32;
    // smem: mean[num_warps] + M2[num_warps] + n[num_warps as int]
    size_t smem = (2 * num_warps) * sizeof(float) + num_warps * sizeof(int);
    kernel_layernorm_v4<<<batch, threads, smem, execution_stream()>>>(d_x, d_gamma, d_beta, d_y, hidden, eps);
    CUDA_CHECK(cudaGetLastError());
}
