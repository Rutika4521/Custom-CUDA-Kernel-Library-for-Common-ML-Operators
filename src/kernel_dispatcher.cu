#include "kernel_dispatcher.cuh"
#include "matmul.cuh"
#include "layernorm.cuh"
#include "softmax.cuh"
#include "shape_kernels.cuh"
#include "benchmark.cuh"
#include "validation.cuh"
#include <mutex>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>
namespace cuda_kernels {
namespace {
std::mutex cache_mutex;
std::unordered_map<std::string,int> cache;
std::string key(const char* op,int a,int b,int c=0) {
    int device=0;CUDA_CHECK(cudaGetDevice(&device));
    return std::string(op)+":"+std::to_string(device)+":"+std::to_string(a)+":"+std::to_string(b)+":"+std::to_string(c);
}
std::string eps_key(float e){uint32_t bits;std::memcpy(&bits,&e,sizeof(bits));return std::to_string(bits);}
int lookup(const std::string& k,int fallback){
    std::lock_guard<std::mutex> guard(cache_mutex);auto i=cache.find(k);return i==cache.end()?fallback:i->second;
}
void save(const std::string& k,int v){std::lock_guard<std::mutex>guard(cache_mutex);cache[k]=v;}
struct Buffer {
    float* p;
    explicit Buffer(const std::vector<float>&v):p(device_alloc_and_copy(v.data(),v.size())){}
    explicit Buffer(size_t n):p(device_alloc<float>(n)){}
    ~Buffer(){device_free(p);}
};
std::vector<float> data(size_t n,unsigned seed,float scale=0.5f){
    std::mt19937 rng(seed);std::uniform_real_distribution<float>d(-scale,scale);std::vector<float>v(n);
    for(float&x:v)x=d(rng);return v;
}
void mm(int v,const float*a,const float*b,float*c,int m,int k,int n,cublasHandle_t handle){
    switch(v){
    case 1:launch_matmul_v1(a,b,c,m,k,n);break;
    case 2:launch_matmul_v2<32>(a,b,c,m,k,n);break;
    case 3:launch_matmul_v3(a,b,c,m,k,n);break;
    case 4:launch_matmul_v4(a,b,c,m,k,n);break;
    case 5:if(handle){launch_matmul_cublas(handle,a,b,c,m,k,n);break;}[[fallthrough]];
    case 6:launch_matmul_shape(a,b,c,m,k,n);break;
    case 7:launch_matmul_v2<16>(a,b,c,m,k,n);break;
    default:
        if(v>=100&&v<100+MATMUL_CONFIG_COUNT){launch_matmul_config(v-100,a,b,c,m,k,n);break;}
        throw std::logic_error("Invalid cached MatMul version");
    }
}
void ln(int v,const float*x,const float*g,const float*b,float*y,int r,int h,float eps){
    switch(v){
    case 1:launch_layernorm_v1(x,g,b,y,r,h,eps);break;
    case 2:launch_layernorm_v2(x,g,b,y,r,h,eps);break;
    case 3:launch_layernorm_v3(x,g,b,y,r,h,eps);break;
    case 4:launch_layernorm_v4(x,g,b,y,r,h,eps);break;
    case 6:launch_layernorm_shape(x,g,b,y,r,h,eps);break;
    default:throw std::logic_error("Invalid cached LayerNorm version");
    }
}
void sm(int v,const float*x,float*y,int r,int c){
    switch(v){
    case 1:launch_softmax_v1(x,y,r,c);break;
    case 2:launch_softmax_v2(x,y,r,c);break;
    case 3:launch_softmax_v3(x,y,r,c);break;
    case 4:launch_softmax_v4(x,y,r,c);break;
    case 6:launch_softmax_shape(x,y,r,c);break;
    default:throw std::logic_error("Invalid cached Softmax version");
    }
}
int tune(const std::vector<int>&versions,std::function<void(int)>launch,float*out,const std::vector<float>&ref){
    BenchmarkConfig cfg;cfg.warmup_iters=10;cfg.bench_iters=30;cfg.verbose=false;
    std::vector<float>check(ref.size());int best=0;float best_ms=INFINITY;
    for(int v:versions){
        CUDA_CHECK(cudaMemset(out,0xff,check.size()*sizeof(float)));launch(v);
        CUDA_CHECK(cudaDeviceSynchronize());device_to_host(check.data(),out,check.size());
        if(!check_correctness(ref.data(),check.data(),check.size(),1e-4f,"tuning candidate",1e-4f).passed)continue;
        std::vector<float> medians;
        for(int trial=0;trial<3;++trial)medians.push_back(run_benchmark("candidate",[&]{launch(v);},cfg).med_ms);
        std::sort(medians.begin(),medians.end());float ms=medians[1];
        printf("  V%d: %.6f ms median of three trials\n",v,ms);
        if(ms<best_ms){best_ms=ms;best=v;}
    }
    if(!best)throw std::runtime_error("No correct tuning candidate.");
    printf("  Best: V%d (%.6f ms)\n",best,best_ms);return best;
}
}
void matmul(const float*a,const float*b,float*c,int m,int k,int n,cublasHandle_t handle){
    validate_matmul(a,b,c,m,k,n);if(!m)return;
    std::string id=key(handle?"matmul_vendor":"matmul_custom",m,k,n);
    int fallback=handle&&(int64_t)m*k*n>=2048ll*2048*2048?5:m<=16?6:3;
    mm(lookup(id,fallback),a,b,c,m,k,n,handle);
}
void layernorm(const float*x,const float*g,const float*b,float*y,int r,int h,float eps){
    validate_layernorm(x,g,b,y,r,h,eps);if(!r)return;
    std::string id=key("layernorm",r,h)+":"+eps_key(eps);
    ln(lookup(id,h<=4096?6:3),x,g,b,y,r,h,eps);
}
void softmax(const float*x,float*y,int r,int c){
    validate_softmax(x,y,r,c);if(!r)return;
    sm(lookup(key("softmax",r,c),c<=4096?6:3),x,y,r,c);
}
void autotune_matmul(int m,int k,int n,cublasHandle_t handle){
    validate_shape(m,k);validate_shape(k,n);validate_shape(m,n);if(!m)return;
    printf("[AutoTune] MatMul %dx%dx%d\n",m,k,n);
    auto a=data((size_t)m*k,1),b=data((size_t)k*n,2);Buffer da(a),db(b),dc((size_t)m*n);
    cublasHandle_t reference_handle=handle;
    if(!reference_handle){CUBLAS_CHECK(cublasCreate(&reference_handle));CUBLAS_CHECK(cublasSetMathMode(reference_handle,CUBLAS_PEDANTIC_MATH));}
    launch_matmul_cublas(reference_handle,da.p,db.p,dc.p,m,k,n);
    std::vector<float>ref((size_t)m*n);device_to_host(ref.data(),dc.p,ref.size());
    if(!handle)CUBLAS_CHECK(cublasDestroy(reference_handle));
    std::vector<int>versions={1,2,3,4,6,7};
    for(int config=0;config<MATMUL_CONFIG_COUNT;++config)
        if(matmul_config_supported(config,m))versions.push_back(100+config);
    if(handle)versions.push_back(5);
    int best=tune(versions,[&](int v){mm(v,da.p,db.p,dc.p,m,k,n,handle);},dc.p,ref);
    save(key(handle?"matmul_vendor":"matmul_custom",m,k,n),best);
}
void autotune_layernorm(int r,int h){
    validate_shape(r,h);if(!r)return;printf("[AutoTune] LayerNorm %dx%d\n",r,h);
    auto x=data((size_t)r*h,3,2.f),g=data(h,4),b=data(h,5);Buffer dx(x),dg(g),db(b),dy((size_t)r*h);
    std::vector<float>ref((size_t)r*h);cpu_layernorm(x.data(),g.data(),b.data(),ref.data(),r,h);
    int best=tune({1,2,3,4,6},[&](int v){ln(v,dx.p,dg.p,db.p,dy.p,r,h,1e-5f);},dy.p,ref);
    save(key("layernorm",r,h)+":"+eps_key(1e-5f),best);
}
void autotune_softmax(int r,int c){
    validate_shape(r,c);if(!r)return;printf("[AutoTune] Softmax %dx%d\n",r,c);
    auto x=data((size_t)r*c,6,3.f);Buffer dx(x),dy((size_t)r*c);std::vector<float>ref((size_t)r*c);
    cpu_softmax(x.data(),ref.data(),r,c);
    int best=tune({1,2,3,4,6},[&](int v){sm(v,dx.p,dy.p,r,c);},dy.p,ref);
    save(key("softmax",r,c),best);
}
}
