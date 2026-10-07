#include "vendor_baselines.h"
#include "cuda_utils.cuh"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef CUDA_KERNELS_HAS_CUDNN
#include <cudnn.h>
static void ck(cudnnStatus_t s,const char* where){
    if(s!=CUDNN_STATUS_SUCCESS)throw std::runtime_error(std::string(where)+": "+cudnnGetErrorString(s));
}
struct VendorBaselines::Impl {
    cudnnHandle_t handle=nullptr;
    cudnnTensorDescriptor_t soft=nullptr;
    std::vector<cudnnBackendDescriptor_t> descriptors;
    cudnnBackendDescriptor_t plan=nullptr;
    void* workspace=nullptr;float eps=1e-5f;std::string label;
    Impl(){ck(cudnnCreate(&handle),"cudnnCreate");ck(cudnnCreateTensorDescriptor(&soft),"tensor");
        label="cuDNN "+std::to_string(cudnnGetVersion());}
    void clear(){
        if(workspace){cudaFree(workspace);workspace=nullptr;}
        for(auto i=descriptors.rbegin();i!=descriptors.rend();++i)cudnnBackendDestroyDescriptor(*i);
        descriptors.clear();plan=nullptr;
    }
    ~Impl(){clear();if(soft)cudnnDestroyTensorDescriptor(soft);if(handle)cudnnDestroy(handle);}
    cudnnBackendDescriptor_t make(cudnnBackendDescriptorType_t type){
        cudnnBackendDescriptor_t d;ck(cudnnBackendCreateDescriptor(type,&d),"create descriptor");
        descriptors.push_back(d);return d;
    }
    static void set(cudnnBackendDescriptor_t d,cudnnBackendAttributeName_t name,
                    cudnnBackendAttributeType_t type,int64_t count,const void* value){
        ck(cudnnBackendSetAttribute(d,name,type,count,value),"set attribute");
    }
    cudnnBackendDescriptor_t tensor(int64_t id,int64_t rows,int64_t cols,bool byvalue=false){
        auto d=make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);cudnnDataType_t dtype=CUDNN_DATA_FLOAT;
        int64_t dims[4]={rows,cols,1,1},strides[4]={cols,1,1,1},alignment=byvalue?4:16;
        set(d,CUDNN_ATTR_TENSOR_DATA_TYPE,CUDNN_TYPE_DATA_TYPE,1,&dtype);
        set(d,CUDNN_ATTR_TENSOR_DIMENSIONS,CUDNN_TYPE_INT64,4,dims);
        set(d,CUDNN_ATTR_TENSOR_STRIDES,CUDNN_TYPE_INT64,4,strides);
        set(d,CUDNN_ATTR_TENSOR_UNIQUE_ID,CUDNN_TYPE_INT64,1,&id);
        set(d,CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,1,&alignment);
        if(byvalue)set(d,CUDNN_ATTR_TENSOR_IS_BY_VALUE,CUDNN_TYPE_BOOLEAN,1,&byvalue);
        ck(cudnnBackendFinalize(d),"finalize tensor");return d;
    }
    void norm(int rows,int cols,float epsilon){
        clear();eps=epsilon;
        auto x=tensor(1,rows,cols),g=tensor(2,1,cols),b=tensor(3,1,cols),
             y=tensor(4,rows,cols),e=tensor(5,1,1,true);
        auto op=make(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
        cudnnBackendNormMode_t mode=CUDNN_LAYER_NORM;
        cudnnBackendNormFwdPhase_t phase=CUDNN_NORM_FWD_INFERENCE;
        set(op,CUDNN_ATTR_OPERATION_NORM_FWD_MODE,CUDNN_TYPE_NORM_MODE,1,&mode);
        set(op,CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,CUDNN_TYPE_NORM_FWD_PHASE,1,&phase);
        set(op,CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&x);
        set(op,CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&g);
        set(op,CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&b);
        set(op,CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&y);
        set(op,CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&e);
        ck(cudnnBackendFinalize(op),"finalize layernorm");
        auto graph=make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        set(graph,CUDNN_ATTR_OPERATIONGRAPH_HANDLE,CUDNN_TYPE_HANDLE,1,&handle);
        set(graph,CUDNN_ATTR_OPERATIONGRAPH_OPS,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&op);
        ck(cudnnBackendFinalize(graph),"finalize graph");
        for(auto hm:{CUDNN_HEUR_MODE_A,CUDNN_HEUR_MODE_B,CUDNN_HEUR_MODE_FALLBACK}){
            auto heur=make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
            set(heur,CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&graph);
            set(heur,CUDNN_ATTR_ENGINEHEUR_MODE,CUDNN_TYPE_HEUR_MODE,1,&hm);
            if(cudnnBackendFinalize(heur)!=CUDNN_STATUS_SUCCESS)continue;
            int64_t count=0;
            if(cudnnBackendGetAttribute(heur,CUDNN_ATTR_ENGINEHEUR_RESULTS,CUDNN_TYPE_BACKEND_DESCRIPTOR,0,&count,nullptr)!=CUDNN_STATUS_SUCCESS)continue;
            std::vector<cudnnBackendDescriptor_t> configs;
            for(int64_t i=0;i<count;++i)configs.push_back(make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
            int64_t got=0;
            if(cudnnBackendGetAttribute(heur,CUDNN_ATTR_ENGINEHEUR_RESULTS,CUDNN_TYPE_BACKEND_DESCRIPTOR,count,&got,configs.data())!=CUDNN_STATUS_SUCCESS)continue;
            for(int64_t i=0;i<got;++i){
                auto p=make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                set(p,CUDNN_ATTR_EXECUTION_PLAN_HANDLE,CUDNN_TYPE_HANDLE,1,&handle);
                set(p,CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&configs[i]);
                if(cudnnBackendFinalize(p)!=CUDNN_STATUS_SUCCESS)continue;
                int64_t bytes=0,n=0;
                ck(cudnnBackendGetAttribute(p,CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,CUDNN_TYPE_INT64,1,&n,&bytes),"workspace size");
                if(bytes)CUDA_CHECK(cudaMalloc(&workspace,bytes));
                plan=p;return;
            }
        }
        throw std::runtime_error("cuDNN has no supported FP32 LayerNorm plan for this shape/device.");
    }
};
VendorBaselines::VendorBaselines():impl(new Impl){}
VendorBaselines::~VendorBaselines()=default;
bool VendorBaselines::available()const{return true;}
const char* VendorBaselines::version()const{return impl->label.c_str();}
void VendorBaselines::prepare_softmax(int r,int c){
    ck(cudnnSetTensor4dDescriptor(impl->soft,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,r,c,1,1),"softmax shape");
}
void VendorBaselines::softmax(const float* x,float* y){
    ck(cudnnSetStream(impl->handle,execution_stream()),"set stream");float a=1,b=0;
    ck(cudnnSoftmaxForward(impl->handle,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_INSTANCE,
                          &a,impl->soft,x,&b,impl->soft,y),"cuDNN Softmax");
}
void VendorBaselines::prepare_layernorm(int r,int c,float eps){impl->norm(r,c,eps);}
void VendorBaselines::layernorm(const float* x,const float* g,const float* b,float* y){
    ck(cudnnSetStream(impl->handle,execution_stream()),"set stream");
    // Binding descriptors are outside device timing; the finalized execution plan is reused.
    int64_t ids[5]={1,2,3,4,5};
    void* ptrs[5]={const_cast<float*>(x),const_cast<float*>(g),const_cast<float*>(b),y,&impl->eps};
    cudnnBackendDescriptor_t pack=nullptr;
    ck(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,&pack),"variant pack");
    try {
        Impl::set(pack,CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,CUDNN_TYPE_INT64,5,ids);
        Impl::set(pack,CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,CUDNN_TYPE_VOID_PTR,5,ptrs);
        Impl::set(pack,CUDNN_ATTR_VARIANT_PACK_WORKSPACE,CUDNN_TYPE_VOID_PTR,1,&impl->workspace);
        ck(cudnnBackendFinalize(pack),"finalize pack");
        ck(cudnnBackendExecute(impl->handle,impl->plan,pack),"cuDNN LayerNorm");
        cudnnBackendDestroyDescriptor(pack);
    }catch(...){cudnnBackendDestroyDescriptor(pack);throw;}
}
#else
struct VendorBaselines::Impl{};
VendorBaselines::VendorBaselines():impl(new Impl){}
VendorBaselines::~VendorBaselines()=default;
bool VendorBaselines::available()const{return false;}
const char* VendorBaselines::version()const{return "cuDNN unavailable";}
void VendorBaselines::prepare_softmax(int,int){throw std::runtime_error("cuDNN unavailable");}
void VendorBaselines::softmax(const float*,float*){throw std::runtime_error("cuDNN unavailable");}
void VendorBaselines::prepare_layernorm(int,int,float){throw std::runtime_error("cuDNN unavailable");}
void VendorBaselines::layernorm(const float*,const float*,const float*,float*){throw std::runtime_error("cuDNN unavailable");}
#endif
