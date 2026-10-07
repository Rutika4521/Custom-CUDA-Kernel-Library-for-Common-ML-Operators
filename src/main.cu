#include <filesystem>
#include <stdexcept>
// =============================================================================
// main.cu — CUDA Kernel Library: Main Driver
// Orchestrates: GPU info → correctness tests → benchmarks → CSV output
// Phases 1–17
// =============================================================================

#include "cuda_utils.cuh"
#include "benchmark.cuh"
#include "matmul.cuh"
#include "layernorm.cuh"
#include "softmax.cuh"
#include "kernel_dispatcher.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

// ─────────────────────────────────────────────────────────────────────────────
// Data generation helpers
// ─────────────────────────────────────────────────────────────────────────────
static void fill_random(float* buf, size_t n, float lo = -1.f, float hi = 1.f,
                         unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

static void fill_ones(float* buf, size_t n) {
    for (size_t i = 0; i < n; ++i) buf[i] = 1.f;
}

static void fill_zeros(float* buf, size_t n) {
    memset(buf, 0, n * sizeof(float));
}

// ─────────────────────────────────────────────────────────────────────────────
// Section separator
// ─────────────────────────────────────────────────────────────────────────────
static void section(const char* title) {
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  %s\n", title);
    printf("══════════════════════════════════════════════════════════════\n");
}

// =============================================================================
// MatMul benchmark suite
// =============================================================================
static void run_matmul_suite(cublasHandle_t cublas,
                              FILE* csv,
                              int M, int K, int N)
{
    char shape[64];
    snprintf(shape, sizeof(shape), "%dx%dx%d", M, K, N);

    printf("\n── MatMul  shape=%s ──\n", shape);

    // Allocate host buffers
    size_t sA = (size_t)M * K, sB = (size_t)K * N, sC = (size_t)M * N;
    std::vector<float> hA(sA), hB(sB), hC_ref(sC), hC_gpu(sC);

    fill_random(hA.data(), sA, -0.5f, 0.5f, 1);
    fill_random(hB.data(), sB, -0.5f, 0.5f, 2);

    // CPU reference (only for small shapes to avoid multi-minute waits)
    bool do_cpu_ref = (M <= 512 && K <= 512 && N <= 512);
    if (do_cpu_ref) {
        cpu_matmul(hA.data(), hB.data(), hC_ref.data(), M, K, N);
    }

    // Device buffers
    float *d_A = device_alloc_and_copy(hA.data(), sA);
    float *d_B = device_alloc_and_copy(hB.data(), sB);
    float *d_C = device_alloc<float>(sC);

    double flops = 2.0 * M * K * N;

    BenchmarkConfig cfg;
    cfg.warmup_iters = 10;
    cfg.bench_iters  = (M >= 2048 ? 30 : 100);

    auto verify = [&](const char* name) {
        if (!do_cpu_ref) return;
        device_to_host(hC_gpu.data(), d_C, sC);
        if (!check_correctness(hC_ref.data(), hC_gpu.data(), sC, 1e-3f, name).passed) throw std::runtime_error("MatMul correctness failed");
    };

    // V1 — Naive
    {
        CUDA_CHECK(cudaMemset(d_C, 0, sC * sizeof(float)));
        launch_matmul_v1(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("MatMul-V1");
        auto r = run_benchmark("Naive(V1)", [&]{ launch_matmul_v1(d_A, d_B, d_C, M, K, N); }, cfg, flops);
        write_csv_row(csv, "MatMul", shape, r);
    }

    // V2 — Tiled 16×16
    {
        CUDA_CHECK(cudaMemset(d_C, 0, sC * sizeof(float)));
        launch_matmul_v2<16>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("MatMul-V2-T16");
        auto r = run_benchmark("Tiled16(V2)", [&]{ launch_matmul_v2<16>(d_A, d_B, d_C, M, K, N); }, cfg, flops);
        write_csv_row(csv, "MatMul", shape, r);
    }

    // V2 — Tiled 32×32
    {
        if (M % 32 == 0 && N % 32 == 0) {
            auto r = run_benchmark("Tiled32(V2)", [&]{ launch_matmul_v2<32>(d_A, d_B, d_C, M, K, N); }, cfg, flops);
            write_csv_row(csv, "MatMul", shape, r);
        }
    }

    // V3 — Register blocked
    {
        CUDA_CHECK(cudaMemset(d_C, 0, sC * sizeof(float)));
        launch_matmul_v3(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("MatMul-V3");
        auto r = run_benchmark("RegBlock(V3)", [&]{ launch_matmul_v3(d_A, d_B, d_C, M, K, N); }, cfg, flops);
        write_csv_row(csv, "MatMul", shape, r);
    }

    // V4 — Vectorized
    {
        CUDA_CHECK(cudaMemset(d_C, 0, sC * sizeof(float)));
        launch_matmul_v4(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("MatMul-V4");
        auto r = run_benchmark("Vec32(V4)", [&]{ launch_matmul_v4(d_A, d_B, d_C, M, K, N); }, cfg, flops);
        write_csv_row(csv, "MatMul", shape, r);
    }

    // cuBLAS baseline
    {
        CUDA_CHECK(cudaMemset(d_C, 0, sC * sizeof(float)));
        launch_matmul_cublas(cublas, d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verify("MatMul-cuBLAS");
        auto r = run_benchmark("cuBLAS", [&]{ launch_matmul_cublas(cublas, d_A, d_B, d_C, M, K, N); }, cfg, flops);
        write_csv_row(csv, "MatMul", shape, r);
    }

    device_free(d_A); device_free(d_B); device_free(d_C);
}

// =============================================================================
// LayerNorm benchmark suite
// =============================================================================
static void run_layernorm_suite(FILE* csv, int batch, int hidden)
{
    char shape[64];
    snprintf(shape, sizeof(shape), "%dx%d", batch, hidden);
    printf("\n── LayerNorm  shape=%s ──\n", shape);

    size_t n = (size_t)batch * hidden;
    std::vector<float> hx(n), hg(hidden), hb(hidden);
    std::vector<float> hy_ref(n), hy_gpu(n);

    fill_random(hx.data(), n, -2.f, 2.f, 10);
    fill_ones(hg.data(), hidden);    // gamma = 1
    fill_zeros(hb.data(), hidden);   // beta  = 0

    cpu_layernorm(hx.data(), hg.data(), hb.data(), hy_ref.data(), batch, hidden);

    float *d_x = device_alloc_and_copy(hx.data(), n);
    float *d_g = device_alloc_and_copy(hg.data(), hidden);
    float *d_b = device_alloc_and_copy(hb.data(), hidden);
    float *d_y = device_alloc<float>(n);

    BenchmarkConfig cfg;
    cfg.warmup_iters = 20;
    cfg.bench_iters  = 100;

    auto verify = [&](const char* name) {
        device_to_host(hy_gpu.data(), d_y, n);
        if (!check_correctness(hy_ref.data(), hy_gpu.data(), n, 1e-4f, name).passed) throw std::runtime_error("LayerNorm correctness failed");
    };

    auto bench_v = [&](int v) {
        CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));
        switch (v) {
            case 1: launch_layernorm_v1(d_x, d_g, d_b, d_y, batch, hidden); break;
            case 2: launch_layernorm_v2(d_x, d_g, d_b, d_y, batch, hidden); break;
            case 3: launch_layernorm_v3(d_x, d_g, d_b, d_y, batch, hidden); break;
            case 4: launch_layernorm_v4(d_x, d_g, d_b, d_y, batch, hidden); break;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        char nm[32]; snprintf(nm, 32, "LN-V%d", v);
        verify(nm);

        std::string label = "LayerNorm-V" + std::to_string(v);
        auto fn = [&]{ switch(v){
            case 1: launch_layernorm_v1(d_x, d_g, d_b, d_y, batch, hidden); break;
            case 2: launch_layernorm_v2(d_x, d_g, d_b, d_y, batch, hidden); break;
            case 3: launch_layernorm_v3(d_x, d_g, d_b, d_y, batch, hidden); break;
            case 4: launch_layernorm_v4(d_x, d_g, d_b, d_y, batch, hidden); break;
        }};
        auto r = run_benchmark(label, fn, cfg);
        write_csv_row(csv, "LayerNorm", shape, r);
    };

    for (int v = 1; v <= 4; ++v) bench_v(v);

    device_free(d_x); device_free(d_g); device_free(d_b); device_free(d_y);
}

// =============================================================================
// Softmax benchmark suite
// =============================================================================
static void run_softmax_suite(FILE* csv, int batch, int cols)
{
    char shape[64];
    snprintf(shape, sizeof(shape), "%dx%d", batch, cols);
    printf("\n── Softmax  shape=%s ──\n", shape);

    size_t n = (size_t)batch * cols;
    std::vector<float> hx(n), hy_ref(n), hy_gpu(n);
    fill_random(hx.data(), n, -3.f, 3.f, 20);
    cpu_softmax(hx.data(), hy_ref.data(), batch, cols);

    float *d_x = device_alloc_and_copy(hx.data(), n);
    float *d_y = device_alloc<float>(n);

    BenchmarkConfig cfg;
    cfg.warmup_iters = 20;
    cfg.bench_iters  = 100;

    auto verify = [&](const char* name) {
        device_to_host(hy_gpu.data(), d_y, n);
        auto res = check_correctness(hy_ref.data(), hy_gpu.data(), n, 1e-4f, name);
        if (!res.passed) throw std::runtime_error("Softmax correctness failed");

        // Also verify row sums ≈ 1.0
        float max_row_sum_err = 0.f;
        for (int b = 0; b < batch; ++b) {
            float rsum = 0.f;
            for (int c = 0; c < cols; ++c) rsum += hy_gpu[b * cols + c];
            max_row_sum_err = fmaxf(max_row_sum_err, fabsf(rsum - 1.f));
        }
        printf("  Row-sum max error: %.2e %s\n",
               max_row_sum_err,
               max_row_sum_err < 1e-4f ? "(OK)" : "(WARN)");
    };

    auto bench_v = [&](int v) {
        CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));
        switch (v) {
            case 1: launch_softmax_v1(d_x, d_y, batch, cols); break;
            case 2: launch_softmax_v2(d_x, d_y, batch, cols); break;
            case 3: launch_softmax_v3(d_x, d_y, batch, cols); break;
            case 4: launch_softmax_v4(d_x, d_y, batch, cols); break;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        char nm[32]; snprintf(nm, 32, "SM-V%d", v);
        verify(nm);

        std::string label = "Softmax-V" + std::to_string(v);
        auto fn = [&]{ switch(v){
            case 1: launch_softmax_v1(d_x, d_y, batch, cols); break;
            case 2: launch_softmax_v2(d_x, d_y, batch, cols); break;
            case 3: launch_softmax_v3(d_x, d_y, batch, cols); break;
            case 4: launch_softmax_v4(d_x, d_y, batch, cols); break;
        }};
        auto r = run_benchmark(label, fn, cfg);
        write_csv_row(csv, "Softmax", shape, r);
    };

    for (int v = 1; v <= 4; ++v) bench_v(v);

    device_free(d_x); device_free(d_y);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv)
{
    try {
    // Phase 1: GPU info
    section("Phase 1 — GPU Environment");
    cuda_init_check();
    CUDA_CHECK(cudaSetDevice(0));
    print_gpu_info(0);

    // cuBLAS handle
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // CSV output
    std::filesystem::create_directories("results");
    FILE* csv = fopen("results/benchmark_results.csv", "w");
    if (!csv) {
        fprintf(stderr, "Warning: cannot open results/benchmark_results.csv\n");
        csv = stdout;  // fallback
    }
    write_csv_header(csv);

    // ── Phase 2: Target shapes ─────────────────────────────────────────────
    // MatMul shapes
    section("Phase 4–6: MatMul Benchmarks");
    struct MM { int M, K, N; };
    std::vector<MM> mm_shapes = {
        {512,  512,  512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        // 4096×4096 may be very slow on V1 — uncomment when desired:
        // {4096, 4096, 4096},
    };
    for (auto& s : mm_shapes)
        run_matmul_suite(cublas, csv, s.M, s.K, s.N);

    // LayerNorm shapes
    section("Phase 7: LayerNorm Benchmarks");
    struct LN { int batch, hidden; };
    std::vector<LN> ln_shapes = {
        {32,  1024},
        {32,  4096},
        {128, 4096},
    };
    for (auto& s : ln_shapes)
        run_layernorm_suite(csv, s.batch, s.hidden);

    // Softmax shapes
    section("Phase 8: Softmax Benchmarks");
    struct SM { int batch, cols; };
    std::vector<SM> sm_shapes = {
        {32,  1024},
        {32,  4096},
        {128, 4096},
    };
    for (auto& s : sm_shapes)
        run_softmax_suite(csv, s.batch, s.cols);

    // ── Phase 14: Auto-tuning demo ─────────────────────────────────────────
    section("Phase 14 — Auto-Tuning Demo");
    cuda_kernels::autotune_matmul(1024, 1024, 1024, cublas);
    cuda_kernels::autotune_layernorm(128, 4096);
    cuda_kernels::autotune_softmax(128, 4096);

    // ── Phase 15: API demo ─────────────────────────────────────────────────
    section("Phase 15 — Public API Demo (Dispatcher)");
    {
        int M = 1024, K = 1024, N = 1024;
        float* d_A = device_alloc<float>((size_t)M * K);
        CUDA_CHECK(cudaMemset(d_A,0,((size_t)M * K)*sizeof(float)));
        float* d_B = device_alloc<float>((size_t)K * N);
        CUDA_CHECK(cudaMemset(d_B,0,((size_t)K * N)*sizeof(float)));
        float* d_C = device_alloc<float>((size_t)M * N);
        CUDA_CHECK(cudaMemset(d_C,0,((size_t)M * N)*sizeof(float)));
        printf("  cuda_kernels::matmul(%d×%d×%d) dispatched to best kernel.\n", M, K, N);
        cuda_kernels::matmul(d_A, d_B, d_C, M, K, N, cublas);
        CUDA_CHECK(cudaDeviceSynchronize());
        device_free(d_A); device_free(d_B); device_free(d_C);

        int B = 128, H = 4096;
        float* dx = device_alloc<float>((size_t)B * H);
        CUDA_CHECK(cudaMemset(dx,0,((size_t)B * H)*sizeof(float)));
        float* dg = device_alloc<float>(H);
        CUDA_CHECK(cudaMemset(dg,0,(H)*sizeof(float)));
        float* db = device_alloc<float>(H);
        CUDA_CHECK(cudaMemset(db,0,(H)*sizeof(float)));
        float* dy = device_alloc<float>((size_t)B * H);
        CUDA_CHECK(cudaMemset(dy,0,((size_t)B * H)*sizeof(float)));
        printf("  cuda_kernels::layernorm(%d×%d) dispatched.\n", B, H);
        cuda_kernels::layernorm(dx, dg, db, dy, B, H);
        CUDA_CHECK(cudaDeviceSynchronize());
        device_free(dx); device_free(dg); device_free(db); device_free(dy);

        float* sx = device_alloc<float>((size_t)B * H);
        CUDA_CHECK(cudaMemset(sx,0,((size_t)B * H)*sizeof(float)));
        float* sy = device_alloc<float>((size_t)B * H);
        CUDA_CHECK(cudaMemset(sy,0,((size_t)B * H)*sizeof(float)));
        printf("  cuda_kernels::softmax(%d×%d) dispatched.\n", B, H);
        cuda_kernels::softmax(sx, sy, B, H);
        CUDA_CHECK(cudaDeviceSynchronize());
        device_free(sx); device_free(sy);
    }

    printf("\n");
    section("Complete — Results written to results/benchmark_results.csv");

    if (csv != stdout) fclose(csv);
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
    } catch (const std::exception& e) { fprintf(stderr,"[ERROR] %s\n",e.what()); return 1; }
}
