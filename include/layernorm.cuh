// =============================================================================
// layernorm.cuh — Layer Normalization Kernel Declarations
// Phases 7
// =============================================================================
#pragma once

#include <cuda_runtime.h>

// CPU Reference (Phase 3)
// Input shape: [batch, hidden]
// gamma/beta are [hidden]
void cpu_layernorm(const float* x, const float* gamma, const float* beta,
                   float* y, int batch, int hidden, float eps = 1e-5f);

// V1 — Shared-memory: one block per row, two-pass (mean then var)
void launch_layernorm_v1(const float* d_x, const float* d_gamma,
                          const float* d_beta, float* d_y,
                          int batch, int hidden, float eps = 1e-5f);

// V2 — Shared-memory: efficient parallel reduction
void launch_layernorm_v2(const float* d_x, const float* d_gamma,
                          const float* d_beta, float* d_y,
                          int batch, int hidden, float eps = 1e-5f);

// V3 — Warp-level primitives (__shfl_down_sync)
void launch_layernorm_v3(const float* d_x, const float* d_gamma,
                          const float* d_beta, float* d_y,
                          int batch, int hidden, float eps = 1e-5f);

// V4 — Vectorized loads + fused mean/variance + minimal gmem traffic
void launch_layernorm_v4(const float* d_x, const float* d_gamma,
                          const float* d_beta, float* d_y,
                          int batch, int hidden, float eps = 1e-5f);
