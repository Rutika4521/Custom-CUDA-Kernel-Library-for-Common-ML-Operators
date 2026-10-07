#pragma once
#include <cuda_runtime.h>
void launch_matmul_shape(const float*,const float*,float*,int,int,int);
void launch_layernorm_shape(const float*,const float*,const float*,float*,int,int,float eps=1e-5f);
void launch_softmax_shape(const float*,float*,int,int);

constexpr int MATMUL_CONFIG_COUNT=11;
const char* matmul_config_name(int config);
bool matmul_config_supported(int config,int m);
void launch_matmul_config(int config,const float*,const float*,float*,int,int,int);
