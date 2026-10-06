// =============================================================================
// benchmark_layernorm.cu — Standalone LayerNorm Benchmark
// Run: ./benchmark_layernorm [batch hidden]
// =============================================================================
#include "layernorm.cuh"
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

int main(int argc, char** argv) {
    cuda_init_check();
    CUDA_CHECK(cudaSetDevice(0));

    int batch = 128, hidden = 4096;
    if (argc == 3) { batch = atoi(argv[1]); hidden = atoi(argv[2]); }

    printf("\n═══════════════════════════════════════════\n");
    printf("  LayerNorm Benchmark: %d × %d\n", batch, hidden);
    printf("═══════════════════════════════════════════\n");

    size_t n = (size_t)batch * hidden;
    std::vector<float> hx(n), hg(hidden, 1.f), hb(hidden, 0.f);
    std::mt19937 rng(7); std::uniform_real_distribution<float> d(-2.f, 2.f);
    for (auto& v : hx) v = d(rng);

    float* d_x = device_alloc_and_copy(hx.data(), n);
    float* d_g = device_alloc_and_copy(hg.data(), hidden);
    float* d_b = device_alloc_and_copy(hb.data(), hidden);
    float* d_y = device_alloc<float>(n);

    BenchmarkConfig cfg; cfg.warmup_iters = 20; cfg.bench_iters = 200;

    printf("\nKernel Results:\n");
    auto r1 = run_benchmark("LN-V1-BasicSMem",    [&]{ launch_layernorm_v1(d_x, d_g, d_b, d_y, batch, hidden); }, cfg);
    auto r2 = run_benchmark("LN-V2-WarpAligned",  [&]{ launch_layernorm_v2(d_x, d_g, d_b, d_y, batch, hidden); }, cfg);
    auto r3 = run_benchmark("LN-V3-ShflPrim",     [&]{ launch_layernorm_v3(d_x, d_g, d_b, d_y, batch, hidden); }, cfg);
    auto r4 = run_benchmark("LN-V4-Welford+Vec4", [&]{ launch_layernorm_v4(d_x, d_g, d_b, d_y, batch, hidden); }, cfg);

    printf("\nSpeedups (vs V1):\n");
    print_speedup(r1, r2);
    print_speedup(r1, r3);
    print_speedup(r1, r4);

    device_free(d_x); device_free(d_g); device_free(d_b); device_free(d_y);
    return 0;
}
