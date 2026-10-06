// =============================================================================
// benchmark_softmax.cu — Standalone Softmax Benchmark
// Run: ./benchmark_softmax [batch cols]
// =============================================================================
#include "softmax.cuh"
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

int main(int argc, char** argv) {
    cuda_init_check();
    CUDA_CHECK(cudaSetDevice(0));

    int batch = 128, cols = 4096;
    if (argc == 3) { batch = atoi(argv[1]); cols = atoi(argv[2]); }

    printf("\n═══════════════════════════════════════════\n");
    printf("  Softmax Benchmark: %d × %d\n", batch, cols);
    printf("═══════════════════════════════════════════\n");

    size_t n = (size_t)batch * cols;
    std::vector<float> hx(n);
    std::mt19937 rng(7); std::uniform_real_distribution<float> d(-3.f, 3.f);
    for (auto& v : hx) v = d(rng);

    float* d_x = device_alloc_and_copy(hx.data(), n);
    float* d_y = device_alloc<float>(n);

    BenchmarkConfig cfg; cfg.warmup_iters = 20; cfg.bench_iters = 200;

    printf("\nKernel Results:\n");
    auto r1 = run_benchmark("SM-V1-BasicSMem",     [&]{ launch_softmax_v1(d_x, d_y, batch, cols); }, cfg);
    auto r2 = run_benchmark("SM-V2-WarpMax",       [&]{ launch_softmax_v2(d_x, d_y, batch, cols); }, cfg);
    auto r3 = run_benchmark("SM-V3-WarpMaxSum",    [&]{ launch_softmax_v3(d_x, d_y, batch, cols); }, cfg);
    auto r4 = run_benchmark("SM-V4-OnlineVec4",    [&]{ launch_softmax_v4(d_x, d_y, batch, cols); }, cfg);

    printf("\nSpeedups (vs V1):\n");
    print_speedup(r1, r2);
    print_speedup(r1, r3);
    print_speedup(r1, r4);

    device_free(d_x); device_free(d_y);
    return 0;
}
