#include "matmul.cuh"
#include "layernorm.cuh"
#include "softmax.cuh"
#include "shape_kernels.cuh"
#include "kernel_dispatcher.cuh"
#include "benchmark.cuh"
#include <limits>
#include <stdexcept>
#include <vector>
#include <random>
#include <functional>

static void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
static void check_rows(const std::vector<float>& y,int rows,int cols) {
    for(int r=0;r<rows;++r){double sum=0;for(int c=0;c<cols;++c){float v=y[r*cols+c];require(std::isfinite(v)&&v>=0,"Invalid probability");sum+=v;}
        require(std::abs(sum-1.)<1e-4,"Softmax row does not sum to one");}
}
int main() {
    try {
        cuda_init_check();
        std::mt19937 rng(42);std::uniform_real_distribution<float> dist(-2.f,2.f);
        cuda_kernels::autotune_layernorm(5,769);
        cuda_kernels::autotune_softmax(5,129);
        cuda_kernels::autotune_matmul(5,33,7);
        for(int cols:{1,3,33,129,768,769,3072,4097})for(int offset:{0,1}) {
            int rows=5,n=rows*cols;
            std::vector<float>x(n),g(cols),b(cols),ref(n),out(n);
            for(float&v:x)v=dist(rng);for(float&v:g)v=dist(rng);for(float&v:b)v=dist(rng);
            float* xb=device_alloc<float>(n+1),*gb=device_alloc<float>(cols+1),*bb=device_alloc<float>(cols+1),*yb=device_alloc<float>(n+1);
            float*dx=xb+offset,*dg=gb+offset,*db=bb+offset,*dy=yb+offset;
            CUDA_CHECK(cudaMemcpy(dx,x.data(),n*sizeof(float),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dg,g.data(),cols*sizeof(float),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(db,b.data(),cols*sizeof(float),cudaMemcpyHostToDevice));
            cpu_layernorm(x.data(),g.data(),b.data(),ref.data(),rows,cols);
            auto ln=[&](const char* name,std::function<void()> fn){
                fn();CUDA_CHECK(cudaDeviceSynchronize());device_to_host(out.data(),dy,n);
                require(check_correctness(ref.data(),out.data(),n,1e-4f,name).passed,"LayerNorm edge case failed");
            };
            ln("LN-V1",[&]{launch_layernorm_v1(dx,dg,db,dy,rows,cols);});
            ln("LN-V2",[&]{launch_layernorm_v2(dx,dg,db,dy,rows,cols);});
            ln("LN-V3",[&]{launch_layernorm_v3(dx,dg,db,dy,rows,cols);});
            ln("LN-V4",[&]{launch_layernorm_v4(dx,dg,db,dy,rows,cols);});
            ln("LN-Shape",[&]{launch_layernorm_shape(dx,dg,db,dy,rows,cols);});
            ln("LN-API",[&]{cuda_kernels::layernorm(dx,dg,db,dy,rows,cols);});
            cpu_softmax(x.data(),ref.data(),rows,cols);
            auto sm=[&](const char* name,std::function<void()> fn){
                fn();CUDA_CHECK(cudaDeviceSynchronize());device_to_host(out.data(),dy,n);
                require(check_correctness(ref.data(),out.data(),n,1e-4f,name).passed,"Softmax edge case failed");
                check_rows(out,rows,cols);
            };
            sm("SM-V1",[&]{launch_softmax_v1(dx,dy,rows,cols);});
            sm("SM-V2",[&]{launch_softmax_v2(dx,dy,rows,cols);});
            sm("SM-V3",[&]{launch_softmax_v3(dx,dy,rows,cols);});
            sm("SM-V4",[&]{launch_softmax_v4(dx,dy,rows,cols);});
            sm("SM-Shape",[&]{launch_softmax_shape(dx,dy,rows,cols);});
            sm("SM-API",[&]{cuda_kernels::softmax(dx,dy,rows,cols);});
            std::fill(x.begin(),x.end(),7.f);cpu_layernorm(x.data(),g.data(),b.data(),ref.data(),rows,cols);
            CUDA_CHECK(cudaMemcpy(dx,x.data(),n*sizeof(float),cudaMemcpyHostToDevice));
            ln("LN-Constant",[&]{launch_layernorm_shape(dx,dg,db,dy,rows,cols);});
            for(int i=0;i<n;++i)x[i]=(i%cols==0)?1000.f:-1000.f;
            cpu_softmax(x.data(),ref.data(),rows,cols);
            CUDA_CHECK(cudaMemcpy(dx,x.data(),n*sizeof(float),cudaMemcpyHostToDevice));
            sm("SM-Large",[&]{launch_softmax_shape(dx,dy,rows,cols);});
            device_free(xb);device_free(gb);device_free(bb);device_free(yb);
        }
        for(auto s:std::vector<std::vector<int>>{{1,3,1},{5,33,7},{17,65,129},{1,768,768},{16,768,768}}){
            int m=s[0],k=s[1],n=s[2];std::vector<float>a(m*k),b(k*n),ref(m*n),out(m*n);
            for(float&v:a)v=dist(rng)*0.25f;for(float&v:b)v=dist(rng)*0.25f;
            cpu_matmul(a.data(),b.data(),ref.data(),m,k,n);
            float*da=device_alloc_and_copy(a.data(),a.size()),*db=device_alloc_and_copy(b.data(),b.size()),*dc=device_alloc<float>(out.size());
            launch_matmul_shape(da,db,dc,m,k,n);device_to_host(out.data(),dc,out.size());
            require(check_correctness(ref.data(),out.data(),out.size(),1e-3f,"MM-Shape").passed,"MatMul shape failed");
            cuda_kernels::matmul(da,db,dc,m,k,n);device_to_host(out.data(),dc,out.size());
            require(check_correctness(ref.data(),out.data(),out.size(),1e-3f,"MM-API").passed,"MatMul dispatcher failed");
            device_free(da);device_free(db);device_free(dc);
        }
        launch_softmax_shape(nullptr,nullptr,0,128);
        launch_layernorm_shape(nullptr,nullptr,nullptr,nullptr,0,768);
        launch_matmul_shape(nullptr,nullptr,nullptr,0,768,768);
        bool caught=false;try{launch_softmax_v1(nullptr,nullptr,-1,128);}catch(const std::invalid_argument&){caught=true;}
        require(caught,"Negative dimensions accepted");
        caught=false;try{launch_layernorm_v1(nullptr,nullptr,nullptr,nullptr,0,768,0.f);}catch(const std::invalid_argument&){caught=true;}
        require(caught,"Invalid epsilon accepted");
        float ref=1.f,bad=std::numeric_limits<float>::quiet_NaN();
        require(!check_correctness(&ref,&bad,1,1e-4f,"Expected-NaN-Fail").passed,"NaN incorrectly accepted");
        bad=std::numeric_limits<float>::infinity();
        require(!check_correctness(&bad,&bad,1,1e-4f,"Expected-Inf-Fail").passed,"Infinity incorrectly accepted");
        printf("All edge-case tests PASSED (NaN/Inf rejection above is expected).\n");
        return 0;
    }catch(const std::exception& e){fprintf(stderr,"[TEST FAILED] %s\n",e.what());return 1;}
}
