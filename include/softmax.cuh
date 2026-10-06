// =============================================================================
// softmax.cuh — Softmax Kernel Declarations
// Phase 8
// =============================================================================
#pragma once

#include <cuda_runtime.h>

// CPU Reference (Phase 3) — numerically stable row-wise softmax
// Input/output shape: [batch, cols]
void cpu_softmax(const float* x, float* y, int batch, int cols);

// V1 — Shared-memory: max then exp+sum, then normalize (three passes in shmem)
void launch_softmax_v1(const float* d_x, float* d_y, int batch, int cols);

// V2 — Warp-level reduction for max value
void launch_softmax_v2(const float* d_x, float* d_y, int batch, int cols);

// V3 — Warp-level reduction for both max and sum
void launch_softmax_v3(const float* d_x, float* d_y, int batch, int cols);

// V4 — Vectorized loads + online softmax (single-pass Milakov algorithm)
void launch_softmax_v4(const float* d_x, float* d_y, int batch, int cols);
