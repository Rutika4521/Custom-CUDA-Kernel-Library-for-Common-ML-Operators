// =============================================================================
// matmul.cuh — MatMul/GEMM Kernel Declarations
// Phases 4, 5, 6
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

// -----------------------------------------------------------------------------
// CPU Reference (Phase 3)
// C[M×N] = A[M×K] × B[K×N]   row-major storage
// -----------------------------------------------------------------------------
void cpu_matmul(const float* A, const float* B, float* C,
                int M, int K, int N);

// -----------------------------------------------------------------------------
// Naive MatMul — V1 (Phase 4)
// One thread computes one C[row,col]
// -----------------------------------------------------------------------------
void launch_matmul_v1(const float* d_A, const float* d_B, float* d_C,
                      int M, int K, int N);

// -----------------------------------------------------------------------------
// Tiled Shared-Memory MatMul — V2 (Phase 5)
// Template TILE_SIZE allows 16 or 32 at compile time
// -----------------------------------------------------------------------------
template<int TILE>
void launch_matmul_v2(const float* d_A, const float* d_B, float* d_C,
                      int M, int K, int N);

// Explicit instantiations declared here, defined in matmul.cu
extern template void launch_matmul_v2<16>(const float*, const float*, float*, int, int, int);
extern template void launch_matmul_v2<32>(const float*, const float*, float*, int, int, int);

// -----------------------------------------------------------------------------
// Register-blocking MatMul — V3 (Phase 5)
// Each thread computes THREAD_M × THREAD_N output elements
// -----------------------------------------------------------------------------
void launch_matmul_v3(const float* d_A, const float* d_B, float* d_C,
                      int M, int K, int N);

// -----------------------------------------------------------------------------
// Unrolled tiled MatMul — V4 (Phase 5)
// Uses scalar tile loads and unrolled multiply-add groups
// -----------------------------------------------------------------------------
void launch_matmul_v4(const float* d_A, const float* d_B, float* d_C,
                      int M, int K, int N);

// -----------------------------------------------------------------------------
// cuBLAS GEMM baseline — V5 (Phase 6)
// -----------------------------------------------------------------------------
void launch_matmul_cublas(cublasHandle_t handle,
                           const float* d_A, const float* d_B, float* d_C,
                           int M, int K, int N);
