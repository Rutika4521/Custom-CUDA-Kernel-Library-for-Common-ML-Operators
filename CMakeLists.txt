cmake_minimum_required(VERSION 3.18)

project(cuda_kernel_library CUDA CXX)

# ── CUDA / C++ standard ───────────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── GPU architecture ─────────────────────────────────────────────────────────
# Change this to match your GPU:
#   RTX 30xx (Ampere)      → 86
#   RTX 40xx (Ada)         → 89
#   RTX 50xx (Blackwell)   → 120   (requires CUDA 12.4+)
#   A100                   → 80
#   H100                   → 90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures")
endif()

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CXX  compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

# ── Include directories ───────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Find cuBLAS via CUDAToolkit ───────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Shared kernel library ─────────────────────────────────────────────────────
add_library(cuda_kernels_lib STATIC
    src/matmul.cu
    src/layernorm.cu
    src/softmax.cu
    src/kernel_dispatcher.cu
)
target_include_directories(cuda_kernels_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cuda_kernels_lib PUBLIC CUDA::cublas CUDA::cudart)
set_target_properties(cuda_kernels_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(cuda_kernel_library src/main.cu)
target_link_libraries(cuda_kernel_library PRIVATE cuda_kernels_lib)
set_target_properties(cuda_kernel_library PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ── Test executables ──────────────────────────────────────────────────────────
foreach(test_name test_matmul test_layernorm test_softmax)
    add_executable(${test_name} tests/${test_name}.cu)
    target_link_libraries(${test_name} PRIVATE cuda_kernels_lib)
    set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

# ── Standalone benchmark executables ─────────────────────────────────────────
foreach(bench_name benchmark_matmul benchmark_layernorm benchmark_softmax)
    add_executable(${bench_name} benchmarks/${bench_name}.cu)
    target_link_libraries(${bench_name} PRIVATE cuda_kernels_lib)
    set_target_properties(${bench_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

# ── Ensure results directory exists at build time ────────────────────────────
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/results)

# ── Convenience target: run all tests ────────────────────────────────────────
add_custom_target(run_tests
    COMMAND test_matmul
    COMMAND test_layernorm
    COMMAND test_softmax
    DEPENDS test_matmul test_layernorm test_softmax
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all correctness tests"
)

# ── CUDA compiler flags ────────────────────────────────────────────────────────
# --use_fast_math : enables hardware intrinsics (rsqrtf, expf, etc.)
# -O3             : maximum host-side optimisation
# -lineinfo       : preserves source line info for Nsight Compute
# --ptxas-options=-v : prints register/smem usage per kernel
target_compile_options(cuda_kernels_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        -lineinfo
        --ptxas-options=-v
    >
)
