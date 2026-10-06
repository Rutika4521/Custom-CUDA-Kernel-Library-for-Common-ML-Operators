// =============================================================================
// test_softmax.cu — Softmax Correctness Tests (Phase 9)
// =============================================================================
#include "softmax.cuh"
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

static bool test_softmax_shape(int batch, int cols, bool verbose = true) {
    if (verbose) printf("\n[TEST] Softmax %d×%d\n", batch, cols);

    size_t n = (size_t)batch * cols;
    std::vector<float> hx(n), hy_ref(n), hy_gpu(n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);
    for (auto& v : hx) v = dist(rng);

    cpu_softmax(hx.data(), hy_ref.data(), batch, cols);

    float* d_x = device_alloc_and_copy(hx.data(), n);
    float* d_y = device_alloc<float>(n);

    bool all_pass = true;
    float tol = 1e-4f;

    auto check_row_sums = [&]() {
        float max_err = 0.f;
        for (int b = 0; b < batch; ++b) {
            float s = 0.f;
            for (int c = 0; c < cols; ++c) s += hy_gpu[b * cols + c];
            max_err = fmaxf(max_err, fabsf(s - 1.f));
        }
        printf("    Row-sum max error: %.2e %s\n",
               max_err, max_err < 1e-4f ? "(OK)" : "(WARN)");
        return max_err < 1e-4f;
    };

    auto run_and_check = [&](const char* name, auto fn) {
        CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        device_to_host(hy_gpu.data(), d_y, n);
        auto r = check_correctness(hy_ref.data(), hy_gpu.data(), n, tol, name);
        bool rs_ok = check_row_sums();
        if (!r.passed || !rs_ok) all_pass = false;
    };

    run_and_check("SM-V1", [&]{ launch_softmax_v1(d_x, d_y, batch, cols); });
    run_and_check("SM-V2", [&]{ launch_softmax_v2(d_x, d_y, batch, cols); });
    run_and_check("SM-V3", [&]{ launch_softmax_v3(d_x, d_y, batch, cols); });
    if (cols % 4 == 0)
        run_and_check("SM-V4", [&]{ launch_softmax_v4(d_x, d_y, batch, cols); });

    device_free(d_x); device_free(d_y);
    return all_pass;
}

int main() {
    cuda_init_check();
    CUDA_CHECK(cudaSetDevice(0));

    printf("╔══════════════════════════════════════╗\n");
    printf("║   Softmax Correctness Test Suite     ║\n");
    printf("╚══════════════════════════════════════╝\n");

    bool all_ok = true;
    all_ok &= test_softmax_shape(1,   128);
    all_ok &= test_softmax_shape(32,  256);
    all_ok &= test_softmax_shape(32,  1024);
    all_ok &= test_softmax_shape(32,  4096);
    all_ok &= test_softmax_shape(128, 4096);

    printf("\n%s\n", all_ok
           ? "\033[32m✓ All Softmax correctness tests PASSED\033[0m"
           : "\033[31m✗ Some Softmax correctness tests FAILED\033[0m");
    return all_ok ? 0 : 1;
}
