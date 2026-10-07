#pragma once
#include <memory>
class VendorBaselines {
public:
    VendorBaselines();~VendorBaselines();
    bool available() const;
    const char* version() const;
    void prepare_softmax(int rows,int cols);
    void softmax(const float* x,float* y);
    void prepare_layernorm(int rows,int cols,float eps=1e-5f);
    void layernorm(const float* x,const float* gamma,const float* beta,float* y);
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
