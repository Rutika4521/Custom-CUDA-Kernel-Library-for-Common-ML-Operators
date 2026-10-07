// =============================================================================
// kernel_dispatcher.cuh — Shape-Aware Kernel Selection (Phase 13)
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuda_kernels {

// -----------------------------------------------------------------------------
// MatMul public API: C[M×N] = A[M×K] × B[K×N]
// The dispatcher selects the best known kernel for the given shape.
// -----------------------------------------------------------------------------
void matmul(const float* d_A, const float* d_B, float* d_C,
            int M, int K, int N,
            cublasHandle_t cublas_handle = nullptr);

// -----------------------------------------------------------------------------
// LayerNorm public API
// Input x: [batch × hidden], gamma/beta: [hidden]
// -----------------------------------------------------------------------------
void layernorm(const float* d_x, const float* d_gamma, const float* d_beta,
               float* d_y, int batch, int hidden, float eps = 1e-5f);

// -----------------------------------------------------------------------------
// Softmax public API — row-wise over [batch × cols]
// -----------------------------------------------------------------------------
void softmax(const float* d_x, float* d_y, int batch, int cols);

// -----------------------------------------------------------------------------
// Auto-tuner: runs a mini-sweep and saves the fastest config for a shape.
// Call once during initialisation; results are cached in memory.
// -----------------------------------------------------------------------------
void autotune_matmul(int M, int K, int N, cublasHandle_t cublas_handle = nullptr);
void autotune_layernorm(int batch, int hidden);
void autotune_softmax(int batch, int cols);

}  // namespace cuda_kernels
