#pragma once
#include <cmath>
#include <climits>
#include <cstdint>
#include <stdexcept>
inline void validate_shape(int rows, int cols) {
    if (rows < 0 || cols <= 0 || (int64_t)rows * cols > INT_MAX)
        throw std::invalid_argument("Invalid shape: rows >= 0, cols > 0, element count <= INT_MAX required.");
}
inline void validate_matmul(const float* a,const float* b,float* c,int m,int k,int n) {
    validate_shape(m,k); validate_shape(k,n); validate_shape(m,n);
    if(m && (!a||!b||!c)) throw std::invalid_argument("Null MatMul buffer.");
}
inline void validate_layernorm(const float* x,const float* g,const float* b,float* y,int r,int h,float e) {
    validate_shape(r,h);
    if(!(e>0.f)||!std::isfinite(e)) throw std::invalid_argument("Epsilon must be finite and positive.");
    if(r && (!x||!g||!b||!y)) throw std::invalid_argument("Null LayerNorm buffer.");
}
inline void validate_softmax(const float* x,float* y,int r,int n) {
    validate_shape(r,n);
    if(r && (!x||!y)) throw std::invalid_argument("Null Softmax buffer.");
}
inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p)&15u)==0; }
