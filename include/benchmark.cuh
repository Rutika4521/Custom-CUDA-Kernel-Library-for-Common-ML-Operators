// =============================================================================
// benchmark.cuh — Reusable CUDA Benchmark Framework
// Phase 10: Benchmark Framework
// =============================================================================
#pragma once

#include "cuda_utils.cuh"
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <string>
#include <cstdio>
#include <cmath>

// -----------------------------------------------------------------------------
// BenchmarkConfig — controls warm-up and measurement iterations
// -----------------------------------------------------------------------------
struct BenchmarkConfig {
    int warmup_iters  = 20;   // warm-up runs (not timed)
    int bench_iters   = 100;  // measured runs
    bool verbose      = true;
};

// -----------------------------------------------------------------------------
// BenchmarkResult — returned for every benchmark run
// -----------------------------------------------------------------------------
struct BenchmarkResult {
    std::string name;
    float avg_ms   = 0.f;
    float min_ms   = 0.f;
    float med_ms   = 0.f;
    float max_ms   = 0.f;
    float tflops   = 0.f;   // for matmul
    double gflops_total = 0.0; // raw FLOPs / time
};

// -----------------------------------------------------------------------------
// Core benchmark runner
// Accepts any callable: void kernel_fn()  (captures everything by reference)
// -----------------------------------------------------------------------------
inline BenchmarkResult run_benchmark(
        const std::string& name,
        std::function<void()> kernel_fn,
        const BenchmarkConfig& cfg = BenchmarkConfig(),
        double flops = 0.0)
{
    CudaTimer timer;
    std::vector<float> times;
    times.reserve(cfg.bench_iters);

    // ── Warm-up ─────────────────────────────────────────────────────────────
    for (int i = 0; i < cfg.warmup_iters; ++i) {
        kernel_fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Timed iterations ─────────────────────────────────────────────────────
    for (int i = 0; i < cfg.bench_iters; ++i) {
        timer.begin();
        kernel_fn();
        timer.end();
        times.push_back(timer.elapsed_ms());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Statistics ───────────────────────────────────────────────────────────
    std::sort(times.begin(), times.end());
    float avg = std::accumulate(times.begin(), times.end(), 0.f) / times.size();
    float mn  = times.front();
    float med = times[times.size() / 2];
    float mx  = times.back();

    BenchmarkResult r;
    r.name   = name;
    r.avg_ms = avg;
    r.min_ms = mn;
    r.med_ms = med;
    r.max_ms = mx;

    if (flops > 0.0) {
        // TFLOPS = FLOPs / (time_s * 1e12)
        double time_s = avg / 1000.0;
        r.tflops       = (float)(flops / (time_s * 1e12));
        r.gflops_total = flops / 1e9;
    }

    if (cfg.verbose) {
        printf("  %-40s  avg=%7.3f ms  min=%7.3f ms  med=%7.3f ms",
               name.c_str(), avg, mn, med);
        if (flops > 0.0) printf("  %.3f TFLOPS", r.tflops);
        printf("\n");
    }
    return r;
}

// -----------------------------------------------------------------------------
// Correctness checker (CPU vs GPU)
// -----------------------------------------------------------------------------
struct CorrectnessResult {
    bool   passed;
    float  max_abs_err;
    float  max_rel_err;
    double avg_abs_err;
};

inline CorrectnessResult check_correctness(
        const float* ref,       // CPU reference output
        const float* gpu,       // GPU result (host buffer)
        size_t       n,
        float        abs_tol = 1e-4f,
        const char*  op_name = "operator")
{
    float  max_abs = 0.f, max_rel = 0.f;
    double sum_abs = 0.0;

    for (size_t i = 0; i < n; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        float rel  = (fabsf(ref[i]) > 1e-8f) ? diff / fabsf(ref[i]) : diff;
        max_abs = fmaxf(max_abs, diff);
        max_rel = fmaxf(max_rel, rel);
        sum_abs += diff;
    }
    bool passed = (max_abs <= abs_tol);

    printf("  Correctness %-12s : %s  (max_abs=%.2e, avg_abs=%.2e, tol=%.2e)\n",
           op_name,
           passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m",
           max_abs, sum_abs / n, abs_tol);

    return { passed, max_abs, max_rel, sum_abs / (double)n };
}

// -----------------------------------------------------------------------------
// Print comparison table row (speedup between two results)
// -----------------------------------------------------------------------------
inline void print_speedup(const BenchmarkResult& baseline,
                           const BenchmarkResult& optimized) {
    float speedup = baseline.avg_ms / optimized.avg_ms;
    printf("  Speedup  %-20s → %-20s : %.2f×\n",
           baseline.name.c_str(), optimized.name.c_str(), speedup);
}

// -----------------------------------------------------------------------------
// CSV writer
// -----------------------------------------------------------------------------
inline void write_csv_header(FILE* f) {
    fprintf(f, "operator,shape,kernel,avg_ms,min_ms,med_ms,tflops,speedup_vs_naive\n");
}

inline void write_csv_row(FILE* f,
                           const char* op, const char* shape,
                           const BenchmarkResult& r,
                           float speedup_vs_naive = 1.f) {
    fprintf(f, "%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            op, shape, r.name.c_str(),
            r.avg_ms, r.min_ms, r.med_ms, r.tflops, speedup_vs_naive);
}
