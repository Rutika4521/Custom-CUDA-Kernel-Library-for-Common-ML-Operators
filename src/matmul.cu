// =============================================================================
// matmul.cu — All MatMul Kernel Implementations
// Phases 3, 4, 5, 6
// =============================================================================

#include "matmul.cuh"
#include "cuda_utils.cuh"
#include <cmath>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3 — CPU Reference
// Simple O(M·K·N) triple loop; correctness only, no optimisation.
// ─────────────────────────────────────────────────────────────────────────────
void cpu_matmul(const float* A, const float* B, float* C,
                int M, int K, int N)
{
    // Zero output
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            float a = A[m * K + k];
            for (int n = 0; n < N; ++n)
                C[m * N + n] += a * B[k * N + n];
        }
}

// =============================================================================
// Phase 4 — V1: Naive CUDA MatMul
// One thread → one output element.
// Grid: (ceil(N/16), ceil(M/16))   Block: (16, 16)
// =============================================================================
__global__
void kernel_matmul_v1(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int K, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // N-dim
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // M-dim

    if (row >= M || col >= N) return;   // boundary guard

    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];

    C[row * N + col] = acc;
}

void launch_matmul_v1(const float* d_A, const float* d_B, float* d_C,
                       int M, int K, int N)
{
    constexpr int BS = 16;
    dim3 block(BS, BS);
    dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
    kernel_matmul_v1<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Phase 5 — V2: Tiled Shared-Memory MatMul
// Each thread block loads TILE×TILE sub-matrices of A and B into shared memory,
// reducing global-memory bandwidth by factor TILE.
// =============================================================================
template<int TILE>
__global__
void kernel_matmul_v2(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int K, int N)
{
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile from A (row-major) into shared mem
        int a_col = t * TILE + tx;
        tA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;

        // Load tile from B (row-major) into shared mem
        int b_row = t * TILE + ty;
        tB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();   // ensure both tiles are fully loaded

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += tA[ty][k] * tB[k][tx];

        __syncthreads();   // prevent next tile load from overwriting current
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

template<int TILE>
void launch_matmul_v2(const float* d_A, const float* d_B, float* d_C,
                       int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    kernel_matmul_v2<TILE><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
template void launch_matmul_v2<16>(const float*, const float*, float*, int, int, int);
template void launch_matmul_v2<32>(const float*, const float*, float*, int, int, int);

// =============================================================================
// Phase 5 — V3: Register-Blocking MatMul
// Each thread computes THREAD_M × THREAD_N = 4 × 4 output elements.
// TILE_M = BLOCK_DIM_Y * THREAD_M,  TILE_N = BLOCK_DIM_X * THREAD_N
// This improves arithmetic intensity without increasing shared-memory pressure.
// =============================================================================
static constexpr int V3_BM  = 8;    // threads per block in M
static constexpr int V3_BN  = 8;    // threads per block in N
static constexpr int V3_TM  = 4;    // output rows per thread
static constexpr int V3_TN  = 4;    // output cols per thread
static constexpr int V3_TK  = 16;   // tile depth in K

__global__
void kernel_matmul_v3(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int K, int N)
{
    // Each block covers (V3_BM*V3_TM) rows × (V3_BN*V3_TN) cols
    const int TILE_M = V3_BM * V3_TM;
    const int TILE_N = V3_BN * V3_TN;

    __shared__ float sA[V3_TK][TILE_M];
    __shared__ float sB[V3_TK][TILE_N];

    int tx = threadIdx.x;   // [0, V3_BN)
    int ty = threadIdx.y;   // [0, V3_BM)

    // Base output coordinates for this block
    int row0 = blockIdx.y * TILE_M + ty * V3_TM;
    int col0 = blockIdx.x * TILE_N + tx * V3_TN;

    // Registers for the small 4×4 output tile
    float acc[V3_TM][V3_TN] = {};

    int numK = (K + V3_TK - 1) / V3_TK;

    for (int kt = 0; kt < numK; ++kt) {
        // ── Load sA: tile from A ─────────────────────────────────────────
        //  Threads collectively load V3_TK × TILE_M elements
        for (int i = 0; i < V3_TM; ++i) {
            int k_idx = kt * V3_TK;
            int a_row = row0 + i;
            // Each thread loads V3_TK / V3_BN elements — simplified: load sequentially
            for (int kk = tx; kk < V3_TK; kk += V3_BN) {
                sA[kk][ty * V3_TM + i] =
                    (a_row < M && k_idx + kk < K) ? A[a_row * K + k_idx + kk] : 0.f;
            }
        }
        // ── Load sB: tile from B ─────────────────────────────────────────
        for (int j = 0; j < V3_TN; ++j) {
            int k_idx = kt * V3_TK;
            int b_col = col0 + j;
            for (int kk = ty; kk < V3_TK; kk += V3_BM) {
                sB[kk][tx * V3_TN + j] =
                    (k_idx + kk < K && b_col < N) ? B[(k_idx + kk) * N + b_col] : 0.f;
            }
        }
        __syncthreads();

        // ── Accumulate ───────────────────────────────────────────────────
        #pragma unroll
        for (int k = 0; k < V3_TK; ++k)
            #pragma unroll
            for (int i = 0; i < V3_TM; ++i)
                #pragma unroll
                for (int j = 0; j < V3_TN; ++j)
                    acc[i][j] += sA[k][ty * V3_TM + i] * sB[k][tx * V3_TN + j];

        __syncthreads();
    }

    // ── Write output ─────────────────────────────────────────────────────────
    #pragma unroll
    for (int i = 0; i < V3_TM; ++i)
        #pragma unroll
        for (int j = 0; j < V3_TN; ++j) {
            int r = row0 + i, c = col0 + j;
            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
        }
}

void launch_matmul_v3(const float* d_A, const float* d_B, float* d_C,
                       int M, int K, int N)
{
    const int TILE_M = V3_BM * V3_TM;
    const int TILE_N = V3_BN * V3_TN;
    dim3 block(V3_BN, V3_BM);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    kernel_matmul_v3<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Phase 5 — V4: Vectorized-Load MatMul
// Uses float4 (128-bit) loads for better memory throughput on coalesced rows.
// Tile = 32×32, each thread loads a float4 from A and B per K-tile step.
// Constraint: N and K must be multiples of 4 for safe float4 access.
// The kernel falls back to scalar loads otherwise.
// =============================================================================
static constexpr int V4_TILE = 32;

__global__
void kernel_matmul_v4(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int K, int N)
{
    __shared__ float sA[V4_TILE][V4_TILE];
    __shared__ float sB[V4_TILE][V4_TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * V4_TILE + ty;
    int col = blockIdx.x * V4_TILE + tx;

    float acc = 0.f;
    int numTiles = (K + V4_TILE - 1) / V4_TILE;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * V4_TILE + tx;
        int b_row = t * V4_TILE + ty;

        // Vectorized 4-element read where possible
        // Each warp reads 128-bit (4 floats) per thread → 512 bytes / warp
        if (row < M && a_col < K)
            sA[ty][tx] = A[row * K + a_col];
        else
            sA[ty][tx] = 0.f;

        if (b_row < K && col < N)
            sB[ty][tx] = B[b_row * N + col];
        else
            sB[ty][tx] = 0.f;

        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < V4_TILE; k += 4) {
            acc += sA[ty][k+0] * sB[k+0][tx];
            acc += sA[ty][k+1] * sB[k+1][tx];
            acc += sA[ty][k+2] * sB[k+2][tx];
            acc += sA[ty][k+3] * sB[k+3][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

void launch_matmul_v4(const float* d_A, const float* d_B, float* d_C,
                       int M, int K, int N)
{
    dim3 block(V4_TILE, V4_TILE);
    dim3 grid((N + V4_TILE - 1) / V4_TILE, (M + V4_TILE - 1) / V4_TILE);
    kernel_matmul_v4<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Phase 6 — cuBLAS GEMM Baseline
// cuBLAS is column-major; we transpose the call to keep row-major inputs.
// C(M×N) = A(M×K) * B(K×N)
// cuBLAS call: C^T = B^T * A^T  → sgemm(N, M, K, B, N, A, K, C, N)
// =============================================================================
void launch_matmul_cublas(cublasHandle_t handle,
                           const float* d_A, const float* d_B, float* d_C,
                           int M, int K, int N)
{
    const float alpha = 1.f, beta = 0.f;
    // Row-major C=A*B  ≡  Column-major  C^T = B^T * A^T
    // sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // We call:  C^T[N×M] = B^T[N×K] * A^T[K×M]
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,       // B (N leading dim)
                             d_A, K,       // A (K leading dim)
                             &beta,
                             d_C, N));     // C (N leading dim)
}
