// =============================================================================
// cuda_utils.cuh — CUDA Error Checking & GPU Info Utilities
// Phase 1: Environment and GPU Information
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// -----------------------------------------------------------------------------
// Error-checking macros
// -----------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "[cuBLAS ERROR] %s:%d  status=%d\n",               \
                    __FILE__, __LINE__, (int)status);                           \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Safe wrappers that print but do not abort (for optional checks)
inline bool cuda_check_soft(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA WARN] %s:%d  %s\n",
                file, line, cudaGetErrorString(err));
        return false;
    }
    return true;
}
#define CUDA_CHECK_SOFT(call) cuda_check_soft((call), __FILE__, __LINE__)

// -----------------------------------------------------------------------------
// GPU device query and display (Phase 1)
// -----------------------------------------------------------------------------

/// Returns the number of available CUDA-capable devices.
/// Calls exit() if none are found.
inline int cuda_init_check() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr,
                "[FATAL] No CUDA-capable GPU found. "
                "This library requires at least one NVIDIA GPU.\n");
        exit(EXIT_FAILURE);
    }
    return device_count;
}

/// Prints detailed properties of a specific CUDA device.
inline void print_gpu_info(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              GPU Device Information (Device %d)              ║\n", device_id);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Name                    : %-35s║\n", prop.name);
    printf("║  Compute Capability      : %d.%-33d║\n",
           prop.major, prop.minor);
    printf("║  Total Global Memory     : %-5.2f GB                         ║\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("║  Shared Mem / Block      : %-6zu KB                        ║\n",
           prop.sharedMemPerBlock / 1024);
    printf("║  Max Threads / Block     : %-35d║\n", prop.maxThreadsPerBlock);
    printf("║  Warp Size               : %-35d║\n", prop.warpSize);
    printf("║  Max Grid Dim X          : %-35d║\n", prop.maxGridSize[0]);
    printf("║  Max Grid Dim Y          : %-35d║\n", prop.maxGridSize[1]);
    printf("║  Max Grid Dim Z          : %-35d║\n", prop.maxGridSize[2]);
    printf("║  Multiprocessors (SMs)   : %-35d║\n", prop.multiProcessorCount);
    printf("║  Max Threads / SM        : %-35d║\n", prop.maxThreadsPerMultiProcessor);
    printf("║  L2 Cache Size           : %-5.2f MB                         ║\n",
           prop.l2CacheSize / (1024.0 * 1024.0));
    printf("║  Memory Bus Width        : %-5d bits                        ║\n",
           prop.memoryBusWidth);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/// Prints a summary of all available CUDA devices.
inline void print_all_gpus() {
    int n = cuda_init_check();
    printf("Found %d CUDA device(s).\n", n);
    for (int i = 0; i < n; ++i) print_gpu_info(i);
}

// -----------------------------------------------------------------------------
// CUDA Event-based timer utility
// -----------------------------------------------------------------------------

struct CudaTimer {
    cudaEvent_t start, stop;

    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    void end()   { CUDA_CHECK(cudaEventRecord(stop));  }
    float elapsed_ms() {
        float ms = 0.f;
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// -----------------------------------------------------------------------------
// Aligned device memory allocation helpers
// -----------------------------------------------------------------------------

template<typename T>
inline T* device_alloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
inline void device_free(T* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

template<typename T>
inline T* device_alloc_and_copy(const T* host_src, size_t count) {
    T* d_ptr = device_alloc<T>(count);
    CUDA_CHECK(cudaMemcpy(d_ptr, host_src, count * sizeof(T),
                          cudaMemcpyHostToDevice));
    return d_ptr;
}

template<typename T>
inline void device_to_host(T* host_dst, const T* dev_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(host_dst, dev_src, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
}
