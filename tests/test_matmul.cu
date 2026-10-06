// =============================================================================
// test_matmul.cu — MatMul Correctness Tests (Phase 9)
// =============================================================================
#include "matmul.cuh"
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

static bool test_matmul_shape(int M, int K, int N, bool verbose = true) {
    if (verbose) printf("\n[TEST] MatMul %d×%d×%d\n", M, K, N);

    size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;
    std::vector<float> hA(sA), hB(sB), hC_ref(sC), hC_gpu(sC);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    cpu_matmul(hA.data(), hB.data(), hC_ref.data(), M, K, N);

    float* d_A = device_alloc_and_copy(hA.data(), sA);
    float* d_B = device_alloc_and_copy(hB.data(), sB);
    float* d_C = device_alloc<float>(sC);

    bool all_pass = true;
    float tol = 1e-3f;   // slightly relaxed for floating-point accumulation

    auto run_and_check = [&](const char* name, auto launch_fn) {
        CUDA_CHECK(cudaMemset(d_C, 0, sC * sizeof(float)));
        launch_fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        device_to_host(hC_gpu.data(), d_C, sC);
        auto r = check_correctness(hC_ref.data(), hC_gpu.data(), sC, tol, name);
        if (!r.passed) all_pass = false;
    };

    run_and_check("V1-Naive",    [&]{ launch_matmul_v1(d_A, d_B, d_C, M, K, N); });
    run_and_check("V2-Tile16",   [&]{ launch_matmul_v2<16>(d_A, d_B, d_C, M, K, N); });
    run_and_check("V2-Tile32",   [&]{ launch_matmul_v2<32>(d_A, d_B, d_C, M, K, N); });
    run_and_check("V3-RegBlock", [&]{ launch_matmul_v3(d_A, d_B, d_C, M, K, N); });
    run_and_check("V4-Vec",      [&]{ launch_matmul_v4(d_A, d_B, d_C, M, K, N); });

    device_free(d_A); device_free(d_B); device_free(d_C);
    return all_pass;
}

int main() {
    cuda_init_check();
    CUDA_CHECK(cudaSetDevice(0));

    printf("╔══════════════════════════════════╗\n");
    printf("║   MatMul Correctness Test Suite  ║\n");
    printf("╚══════════════════════════════════╝\n");

    bool all_ok = true;
    // Test multiple shapes including non-power-of-2
    all_ok &= test_matmul_shape(64,   64,  64);
    all_ok &= test_matmul_shape(128, 128, 128);
    all_ok &= test_matmul_shape(256, 256, 256);
    all_ok &= test_matmul_shape(512, 512, 512);
    all_ok &= test_matmul_shape(100, 200, 150);  // non-aligned

    printf("\n%s\n", all_ok
           ? "\033[32m✓ All MatMul correctness tests PASSED\033[0m"
           : "\033[31m✗ Some MatMul correctness tests FAILED\033[0m");
    return all_ok ? 0 : 1;
}
