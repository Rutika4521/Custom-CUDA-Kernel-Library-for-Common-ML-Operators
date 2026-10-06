// =============================================================================
// test_layernorm.cu — LayerNorm Correctness Tests (Phase 9)
// =============================================================================
#include "layernorm.cuh"
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

static bool test_layernorm_shape(int batch, int hidden, bool verbose = true) {
    if (verbose) printf("\n[TEST] LayerNorm %d×%d\n", batch, hidden);

    size_t n = (size_t)batch * hidden;
    std::vector<float> hx(n), hg(hidden), hb(hidden);
    std::vector<float> hy_ref(n), hy_gpu(n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (auto& v : hx) v = dist(rng);
    for (auto& v : hg) v = dist(rng);  // random gamma
    for (auto& v : hb) v = dist(rng);  // random beta

    cpu_layernorm(hx.data(), hg.data(), hb.data(), hy_ref.data(), batch, hidden);

    float* d_x = device_alloc_and_copy(hx.data(), n);
    float* d_g = device_alloc_and_copy(hg.data(), hidden);
    float* d_b = device_alloc_and_copy(hb.data(), hidden);
    float* d_y = device_alloc<float>(n);

    bool all_pass = true;
    float tol = 1e-4f;

    auto run_and_check = [&](const char* name, auto fn) {
        CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        device_to_host(hy_gpu.data(), d_y, n);
        auto r = check_correctness(hy_ref.data(), hy_gpu.data(), n, tol, name);
        if (!r.passed) all_pass = false;
    };

    run_and_check("LN-V1", [&]{ launch_layernorm_v1(d_x, d_g, d_b, d_y, batch, hidden); });
    run_and_check("LN-V2", [&]{ launch_layernorm_v2(d_x, d_g, d_b, d_y, batch, hidden); });
    run_and_check("LN-V3", [&]{ launch_layernorm_v3(d_x, d_g, d_b, d_y, batch, hidden); });
    if (hidden % 4 == 0)
        run_and_check("LN-V4", [&]{ launch_layernorm_v4(d_x, d_g, d_b, d_y, batch, hidden); });

    device_free(d_x); device_free(d_g); device_free(d_b); device_free(d_y);
    return all_pass;
}

int main() {
    cuda_init_check();
    CUDA_CHECK(cudaSetDevice(0));

    printf("╔════════════════════════════════════╗\n");
    printf("║  LayerNorm Correctness Test Suite  ║\n");
    printf("╚════════════════════════════════════╝\n");

    bool all_ok = true;
    all_ok &= test_layernorm_shape(1,   64);
    all_ok &= test_layernorm_shape(32,  256);
    all_ok &= test_layernorm_shape(32,  1024);
    all_ok &= test_layernorm_shape(32,  4096);
    all_ok &= test_layernorm_shape(128, 4096);

    printf("\n%s\n", all_ok
           ? "\033[32m✓ All LayerNorm correctness tests PASSED\033[0m"
           : "\033[31m✗ Some LayerNorm correctness tests FAILED\033[0m");
    return all_ok ? 0 : 1;
}
