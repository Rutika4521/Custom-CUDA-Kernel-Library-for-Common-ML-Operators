// =============================================================================
// kernel_dispatcher.cu — Shape-Aware Kernel Dispatcher (Phases 13, 14, 15)
// =============================================================================

#include "kernel_dispatcher.cuh"
#include "matmul.cuh"
#include "layernorm.cuh"
#include "softmax.cuh"
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include <unordered_map>
#include <string>
#include <cstdio>
#include <algorithm>

namespace cuda_kernels {

// ─────────────────────────────────────────────────────────────────────────────
// Internal tuning caches
// Key: "M_K_N" or "B_H" or "B_C"
// ─────────────────────────────────────────────────────────────────────────────

static std::unordered_map<std::string, int> matmul_tune_cache;
static std::unordered_map<std::string, int> layernorm_tune_cache;
static std::unordered_map<std::string, int> softmax_tune_cache;

static inline std::string matmul_key(int M, int K, int N) {
    return std::to_string(M) + "_" + std::to_string(K) + "_" + std::to_string(N);
}
static inline std::string ln_key(int B, int H) {
    return std::to_string(B) + "_" + std::to_string(H);
}
static inline std::string sm_key(int B, int C) {
    return std::to_string(B) + "_" + std::to_string(C);
}

// ─────────────────────────────────────────────────────────────────────────────
// MatMul dispatcher
// Heuristic rules (can be overridden by auto-tune):
//   Large square (>= 2048):  V4 (32-tile + unrolled)
//   Medium (>= 512):         V2<32> (tiled SMEM)
//   Small:                   V1 (naive, avoids block-size waste)
// ─────────────────────────────────────────────────────────────────────────────
void matmul(const float* d_A, const float* d_B, float* d_C,
            int M, int K, int N, cublasHandle_t cublas_handle)
{
    std::string key = matmul_key(M, K, N);
    int version = 0;

    auto it = matmul_tune_cache.find(key);
    if (it != matmul_tune_cache.end()) {
        version = it->second;
    } else {
        // Shape-based heuristic selection
        long long size = (long long)M * K * N;
        if (size >= (long long)2048 * 2048 * 2048) {
            version = (cublas_handle ? 5 : 4);  // prefer cuBLAS for huge shapes
        } else if (M >= 512 && N >= 512) {
            version = 4;    // vectorized tiled
        } else if (M >= 128 && N >= 128) {
            version = 2;    // shared-mem tiled
        } else {
            version = 1;    // naive (small shapes have little to gain)
        }
    }

    switch (version) {
        case 1: launch_matmul_v1(d_A, d_B, d_C, M, K, N); break;
        case 2: launch_matmul_v2<32>(d_A, d_B, d_C, M, K, N); break;
        case 3: launch_matmul_v3(d_A, d_B, d_C, M, K, N); break;
        case 4: launch_matmul_v4(d_A, d_B, d_C, M, K, N); break;
        case 5:
            if (cublas_handle)
                launch_matmul_cublas(cublas_handle, d_A, d_B, d_C, M, K, N);
            else
                launch_matmul_v4(d_A, d_B, d_C, M, K, N);
            break;
        default: launch_matmul_v2<32>(d_A, d_B, d_C, M, K, N); break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerNorm dispatcher
// hidden < 1024: V3 (warp primitives, fewer threads needed)
// hidden >= 1024 and multiple of 4: V4 (vectorized + Welford)
// otherwise: V2 (reliable warp-aligned reduction)
// ─────────────────────────────────────────────────────────────────────────────
void layernorm(const float* d_x, const float* d_gamma, const float* d_beta,
               float* d_y, int batch, int hidden, float eps)
{
    std::string key = ln_key(batch, hidden);
    int version = 0;

    auto it = layernorm_tune_cache.find(key);
    if (it != layernorm_tune_cache.end()) {
        version = it->second;
    } else {
        if (hidden % 4 == 0 && hidden >= 1024)
            version = 4;
        else if (hidden >= 512)
            version = 3;
        else
            version = 2;
    }

    switch (version) {
        case 1: launch_layernorm_v1(d_x, d_gamma, d_beta, d_y, batch, hidden, eps); break;
        case 2: launch_layernorm_v2(d_x, d_gamma, d_beta, d_y, batch, hidden, eps); break;
        case 3: launch_layernorm_v3(d_x, d_gamma, d_beta, d_y, batch, hidden, eps); break;
        case 4: launch_layernorm_v4(d_x, d_gamma, d_beta, d_y, batch, hidden, eps); break;
        default: launch_layernorm_v4(d_x, d_gamma, d_beta, d_y, batch, hidden, eps); break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax dispatcher
// cols % 4 == 0 and cols >= 512: V4 (online + vectorized)
// cols >= 128: V3 (warp-level both reductions)
// else: V1 (shared-mem, reliable)
// ─────────────────────────────────────────────────────────────────────────────
void softmax(const float* d_x, float* d_y, int batch, int cols)
{
    std::string key = sm_key(batch, cols);
    int version = 0;

    auto it = softmax_tune_cache.find(key);
    if (it != softmax_tune_cache.end()) {
        version = it->second;
    } else {
        if (cols % 4 == 0 && cols >= 512)
            version = 4;
        else if (cols >= 128)
            version = 3;
        else
            version = 1;
    }

    switch (version) {
        case 1: launch_softmax_v1(d_x, d_y, batch, cols); break;
        case 2: launch_softmax_v2(d_x, d_y, batch, cols); break;
        case 3: launch_softmax_v3(d_x, d_y, batch, cols); break;
        case 4: launch_softmax_v4(d_x, d_y, batch, cols); break;
        default: launch_softmax_v4(d_x, d_y, batch, cols); break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-tuner: MatMul
// Tries a small subset of kernel versions and caches the fastest one.
// ─────────────────────────────────────────────────────────────────────────────
void autotune_matmul(int M, int K, int N)
{
    printf("\n[AutoTune] MatMul %d×%d×%d\n", M, K, N);
    BenchmarkConfig cfg;
    cfg.warmup_iters = 5;
    cfg.bench_iters  = 20;
    cfg.verbose      = false;

    float* d_A = device_alloc<float>((size_t)M * K);
    float* d_B = device_alloc<float>((size_t)K * N);
    float* d_C = device_alloc<float>((size_t)M * N);

    struct KV { int ver; float ms; };
    KV best = { 1, 1e9f };

    auto bench_ver = [&](int ver, auto fn) {
        auto r = run_benchmark("v" + std::to_string(ver), fn, cfg);
        printf("  V%d: %.3f ms\n", ver, r.avg_ms);
        if (r.avg_ms < best.ms) best = { ver, r.avg_ms };
    };

    bench_ver(1, [&]{ launch_matmul_v1(d_A, d_B, d_C, M, K, N); });
    bench_ver(2, [&]{ launch_matmul_v2<32>(d_A, d_B, d_C, M, K, N); });
    bench_ver(3, [&]{ launch_matmul_v3(d_A, d_B, d_C, M, K, N); });
    bench_ver(4, [&]{ launch_matmul_v4(d_A, d_B, d_C, M, K, N); });

    printf("  → Best: V%d (%.3f ms)\n", best.ver, best.ms);
    matmul_tune_cache[matmul_key(M, K, N)] = best.ver;

    device_free(d_A); device_free(d_B); device_free(d_C);
}

void autotune_layernorm(int batch, int hidden)
{
    printf("\n[AutoTune] LayerNorm %d×%d\n", batch, hidden);
    BenchmarkConfig cfg;
    cfg.warmup_iters = 5;
    cfg.bench_iters  = 20;
    cfg.verbose      = false;

    float* d_x = device_alloc<float>((size_t)batch * hidden);
    float* d_g = device_alloc<float>(hidden);
    float* d_b = device_alloc<float>(hidden);
    float* d_y = device_alloc<float>((size_t)batch * hidden);

    struct KV { int ver; float ms; };
    KV best = { 1, 1e9f };

    for (int ver = 1; ver <= 4; ++ver) {
        auto fn = [&]{
            switch(ver) {
                case 1: launch_layernorm_v1(d_x, d_g, d_b, d_y, batch, hidden); break;
                case 2: launch_layernorm_v2(d_x, d_g, d_b, d_y, batch, hidden); break;
                case 3: launch_layernorm_v3(d_x, d_g, d_b, d_y, batch, hidden); break;
                case 4: launch_layernorm_v4(d_x, d_g, d_b, d_y, batch, hidden); break;
            }
        };
        auto r = run_benchmark("ln_v" + std::to_string(ver), fn, cfg);
        printf("  V%d: %.3f ms\n", ver, r.avg_ms);
        if (r.avg_ms < best.ms) best = { ver, r.avg_ms };
    }
    printf("  → Best: V%d\n", best.ver);
    layernorm_tune_cache[ln_key(batch, hidden)] = best.ver;

    device_free(d_x); device_free(d_g); device_free(d_b); device_free(d_y);
}

void autotune_softmax(int batch, int cols)
{
    printf("\n[AutoTune] Softmax %d×%d\n", batch, cols);
    BenchmarkConfig cfg;
    cfg.warmup_iters = 5;
    cfg.bench_iters  = 20;
    cfg.verbose      = false;

    float* d_x = device_alloc<float>((size_t)batch * cols);
    float* d_y = device_alloc<float>((size_t)batch * cols);

    struct KV { int ver; float ms; };
    KV best = { 1, 1e9f };

    for (int ver = 1; ver <= 4; ++ver) {
        auto fn = [&]{
            switch(ver) {
                case 1: launch_softmax_v1(d_x, d_y, batch, cols); break;
                case 2: launch_softmax_v2(d_x, d_y, batch, cols); break;
                case 3: launch_softmax_v3(d_x, d_y, batch, cols); break;
                case 4: launch_softmax_v4(d_x, d_y, batch, cols); break;
            }
        };
        auto r = run_benchmark("sm_v" + std::to_string(ver), fn, cfg);
        printf("  V%d: %.3f ms\n", ver, r.avg_ms);
        if (r.avg_ms < best.ms) best = { ver, r.avg_ms };
    }
    printf("  → Best: V%d\n", best.ver);
    softmax_tune_cache[sm_key(batch, cols)] = best.ver;

    device_free(d_x); device_free(d_y);
}

}  // namespace cuda_kernels
