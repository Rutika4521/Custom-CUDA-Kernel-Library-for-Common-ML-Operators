// =============================================================================
// benchmark_matmul.cu — Standalone MatMul Benchmark (Phase 10–17)
// Run: ./benchmark_matmul [M K N]
// =============================================================================
#include "matmul.cuh"
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

int main(int argc, char** argv) {
    cuda_init_check();
    CUDA_CHECK(cudaSetDevice(0));

    int M = 1024, K = 1024, N = 1024;
    if (argc == 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }

    printf("\n═══════════════════════════════════════════\n");
    printf("  MatMul Benchmark: %d × %d × %d\n", M, K, N);
    printf("═══════════════════════════════════════════\n");

    size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;
    std::vector<float> hA(sA), hB(sB);
    std::mt19937 rng(7); std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& v : hA) v = d(rng);
    for (auto& v : hB) v = d(rng);

    float* d_A = device_alloc_and_copy(hA.data(), sA);
    float* d_B = device_alloc_and_copy(hB.data(), sB);
    float* d_C = device_alloc<float>(sC);

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas));
    double flops = 2.0 * M * K * N;

    BenchmarkConfig cfg;
    cfg.warmup_iters = 20;
    cfg.bench_iters  = 100;

    printf("\nKernel Results:\n");
    auto r1 = run_benchmark("V1-Naive",    [&]{ launch_matmul_v1(d_A, d_B, d_C, M, K, N); },           cfg, flops);
    auto r2 = run_benchmark("V2-Tile16",   [&]{ launch_matmul_v2<16>(d_A, d_B, d_C, M, K, N); },       cfg, flops);
    auto r3 = run_benchmark("V2-Tile32",   [&]{ launch_matmul_v2<32>(d_A, d_B, d_C, M, K, N); },       cfg, flops);
    auto r4 = run_benchmark("V3-RegBlock", [&]{ launch_matmul_v3(d_A, d_B, d_C, M, K, N); },           cfg, flops);
    auto r5 = run_benchmark("V4-Vec32",    [&]{ launch_matmul_v4(d_A, d_B, d_C, M, K, N); },           cfg, flops);
    auto rc = run_benchmark("cuBLAS",      [&]{ launch_matmul_cublas(cublas, d_A, d_B, d_C, M, K, N);}, cfg, flops);

    printf("\nSpeedups (vs Naive):\n");
    print_speedup(r1, r2);
    print_speedup(r1, r3);
    print_speedup(r1, r4);
    print_speedup(r1, r5);
    print_speedup(r1, rc);

    printf("\nCustom Best vs cuBLAS:\n");
    BenchmarkResult best = (r5.avg_ms < r4.avg_ms) ? r5 : r4;
    print_speedup(rc, best);

    CUBLAS_CHECK(cublasDestroy(cublas));
    device_free(d_A); device_free(d_B); device_free(d_C);
    return 0;
}
