#include "shape_kernels.cuh"
#include "cuda_utils.cuh"
#include "validation.cuh"
#include "layernorm.cuh"
#include "softmax.cuh"
#include <cfloat>
// Cooperatively loaded tiles with a register patch per thread.
template<int BM,int BN,int BK,int TM,int TN>
__global__ void kernel_matmul_shape(const float* __restrict__ a,const float* __restrict__ b,
                                    float* __restrict__ c,int m,int k,int n) {
    constexpr int NX=BN/TN, NY=BM/TM, NT=NX*NY;
    __shared__ float sa[BK][BM+1], sb[BK][BN];
    int tx=threadIdx.x%NX,ty=threadIdx.x/NX;
    int br=blockIdx.y*BM,bc=blockIdx.x*BN;
    float acc[TM][TN]={};
    for(int base=0;base<k;base+=BK) {
        for(int i=threadIdx.x;i<BM*BK;i+=NT) {
            int r=i/BK,q=i%BK;
            sa[q][r]=br+r<m&&base+q<k?a[(br+r)*k+base+q]:0.f;
        }
        for(int i=threadIdx.x;i<BK*BN;i+=NT) {
            int q=i/BN,col=i%BN;
            sb[q][col]=base+q<k&&bc+col<n?b[(base+q)*n+bc+col]:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int q=0;q<BK;++q) {
            float av[TM],bv[TN];
            #pragma unroll
            for(int i=0;i<TM;++i) av[i]=sa[q][ty*TM+i];
            #pragma unroll
            for(int j=0;j<TN;++j) bv[j]=sb[q][tx+j*NX];
            #pragma unroll
            for(int i=0;i<TM;++i)
                #pragma unroll
                for(int j=0;j<TN;++j) acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<TM;++i)
        #pragma unroll
        for(int j=0;j<TN;++j) {
            int r=br+ty*TM+i,col=bc+tx+j*NX;
            if(r<m&&col<n)c[r*n+col]=acc[i][j];
        }
}

__global__ void kernel_gemv_shape(const float* __restrict__ a,const float* __restrict__ b,
                                  float* __restrict__ c,int k,int n) {
    __shared__ float sums[8][32];
    int lane=threadIdx.x,part=threadIdx.y,col=blockIdx.x*32+lane;
    float sum=0.f;
    if(col<n)for(int q=part;q<k;q+=8)sum=fmaf(a[q],b[q*n+col],sum);
    sums[part][lane]=sum;
    __syncthreads();
    if(part==0&&col<n){sum=0.f;
        #pragma unroll
        for(int q=0;q<8;++q)sum+=sums[q][lane];
        c[col]=sum;
    }
}
void launch_matmul_shape(const float* a,const float* b,float* c,int m,int k,int n) {
    validate_matmul(a,b,c,m,k,n); if(!m)return;
    if(m==1)kernel_gemv_shape<<<(n+31)/32,dim3(32,8),0,execution_stream()>>>(a,b,c,k,n);
    else if(m<=16)kernel_matmul_shape<16,64,32,1,4><<<dim3((n+63)/64,(m+15)/16),256, 0, execution_stream()>>>(a,b,c,m,k,n);
    else kernel_matmul_shape<64,64,16,4,4><<<dim3((n+63)/64,(m+63)/64),256, 0, execution_stream()>>>(a,b,c,m,k,n);
    CUDA_CHECK(cudaGetLastError());
}
__device__ __forceinline__ float shape_warp_sum(float v) {
    #pragma unroll
    for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);
    return __shfl_sync(0xffffffff,v,0);
}
__device__ __forceinline__ float shape_warp_max(float v) {
    #pragma unroll
    for(int d=16;d;d>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,d));
    return __shfl_sync(0xffffffff,v,0);
}
// Four independent warps per block; each caches one entire row in registers.
template<int ITEMS>
__global__ void kernel_layernorm_shape(const float* __restrict__ x,const float* __restrict__ g,
                                       const float* __restrict__ b,float* __restrict__ y,
                                       int rows,int cols,float eps) {
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);
    if(row>=rows)return;
    float values[ITEMS],sum=0.f;
    #pragma unroll
    for(int j=0;j<ITEMS;++j) {
        int col=lane+32*j;
        values[j]=col<cols?x[row*cols+col]:0.f; sum+=values[j];
    }
    float mean=shape_warp_sum(sum)/cols,var=0.f;
    #pragma unroll
    for(int j=0;j<ITEMS;++j)if(lane+32*j<cols){float d=values[j]-mean;var+=d*d;}
    float inv=rsqrtf(shape_warp_sum(var)/cols+eps);
    #pragma unroll
    for(int j=0;j<ITEMS;++j){int col=lane+32*j;if(col<cols)y[row*cols+col]=g[col]*(values[j]-mean)*inv+b[col];}
}

template<int ITEMS>
__global__ void kernel_layernorm_resident(const float* __restrict__ x,const float* __restrict__ g,
                                          const float* __restrict__ b,float* __restrict__ y,
                                          int cols,float eps) {
    __shared__ float partial[4];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,row=blockIdx.x;
    float values[ITEMS],sum=0.f;
    #pragma unroll
    for(int j=0;j<ITEMS;++j){int col=tid+128*j;values[j]=col<cols?x[row*cols+col]:0.f;sum+=values[j];}
    sum=shape_warp_sum(sum);
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    float total=shape_warp_sum(tid<4?partial[tid]:0.f);
    if(tid==0)partial[0]=total;
    __syncthreads();
    float mean=partial[0]/cols;
    __syncthreads();
    float var=0.f;
    #pragma unroll
    for(int j=0;j<ITEMS;++j)if(tid+128*j<cols){float d=values[j]-mean;var+=d*d;}
    var=shape_warp_sum(var);
    if(lane==0)partial[warp]=var;
    __syncthreads();
    total=shape_warp_sum(tid<4?partial[tid]:0.f);
    if(tid==0)partial[0]=total;
    __syncthreads();
    float inv=rsqrtf(partial[0]/cols+eps);
    #pragma unroll
    for(int j=0;j<ITEMS;++j){int col=tid+128*j;if(col<cols)y[row*cols+col]=g[col]*(values[j]-mean)*inv+b[col];}
}
void launch_layernorm_shape(const float* x,const float* g,const float* b,float* y,int r,int h,float e) {
    validate_layernorm(x,g,b,y,r,h,e);if(!r)return;
    if(h<=128)kernel_layernorm_shape<4><<<dim3((r+3)/4),128,0,execution_stream()>>>(x,g,b,y,r,h,e);
    else if(h<=256)kernel_layernorm_shape<8><<<dim3((r+3)/4),128,0,execution_stream()>>>(x,g,b,y,r,h,e);
    else if(h<=768)kernel_layernorm_resident<6><<<r,128,0,execution_stream()>>>(x,g,b,y,h,e);
    else if(h<=1024)kernel_layernorm_resident<8><<<r,128,0,execution_stream()>>>(x,g,b,y,h,e);
    else if(h<=3072)kernel_layernorm_resident<24><<<r,128,0,execution_stream()>>>(x,g,b,y,h,e);
    else if(h<=4096)kernel_layernorm_resident<32><<<r,128,0,execution_stream()>>>(x,g,b,y,h,e);
    else {launch_layernorm_v3(x,g,b,y,r,h,e);return;}
    CUDA_CHECK(cudaGetLastError());
}
template<int ITEMS>
__global__ void kernel_softmax_shape(const float* __restrict__ x,float* __restrict__ y,int rows,int cols) {
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);if(row>=rows)return;
    float values[ITEMS],mx=-FLT_MAX;
    #pragma unroll
    for(int j=0;j<ITEMS;++j){int c=lane+32*j;values[j]=c<cols?x[row*cols+c]:-FLT_MAX;mx=fmaxf(mx,values[j]);}
    mx=shape_warp_max(mx);float sum=0.f;
    #pragma unroll
    for(int j=0;j<ITEMS;++j){values[j]=lane+32*j<cols?__expf(values[j]-mx):0.f;sum+=values[j];}
    float inv=1.f/shape_warp_sum(sum);
    #pragma unroll
    for(int j=0;j<ITEMS;++j){int c=lane+32*j;if(c<cols)y[row*cols+c]=values[j]*inv;}
}
void launch_softmax_shape(const float* x,float* y,int r,int c) {
    validate_softmax(x,y,r,c);if(!r)return;dim3 grid((r+3)/4);
    if(c<=128)kernel_softmax_shape<4><<<grid,128, 0, execution_stream()>>>(x,y,r,c);
    else if(c<=256)kernel_softmax_shape<8><<<grid,128, 0, execution_stream()>>>(x,y,r,c);
    else if(c<=1024)kernel_softmax_shape<32><<<grid,128, 0, execution_stream()>>>(x,y,r,c);
    else if(c<=4096)kernel_softmax_shape<128><<<grid,128, 0, execution_stream()>>>(x,y,r,c);
    else{launch_softmax_v3(x,y,r,c);return;}
    CUDA_CHECK(cudaGetLastError());
}


// Additional configurations for measured tuning. The scalar path preserves odd-width
// and unaligned-buffer support; vectorized accesses are selected only when safe.
template<int BM,int BN,int BK,int TM,int TN,bool VEC>
__global__ void kernel_matmul_vector(const float* __restrict__ a,const float* __restrict__ b,
                                     float* __restrict__ c,int m,int k,int n) {
    constexpr int NX=BN/TN,NT=NX*(BM/TM);
    __shared__ float sa[BK][BM+1];
    __shared__ __align__(16) float sb[BK][BN];
    int tx=threadIdx.x%NX,ty=threadIdx.x/NX;
    int br=blockIdx.y*BM,bc=blockIdx.x*BN;
    float acc[TM][TN]={};
    for(int base=0;base<k;base+=BK){
        if constexpr(VEC){
            #pragma unroll
            for(int i=threadIdx.x;i<BM*(BK/4);i+=NT){
                int r=i/(BK/4),q=(i%(BK/4))*4;
                float4 v={0,0,0,0};
                if(br+r<m&&base+q<k)
                    v=*reinterpret_cast<const float4*>(a+(br+r)*k+base+q);
                sa[q][r]=v.x;sa[q+1][r]=v.y;sa[q+2][r]=v.z;sa[q+3][r]=v.w;
            }
            #pragma unroll
            for(int i=threadIdx.x;i<BK*(BN/4);i+=NT){
                int q=i/(BN/4),col=(i%(BN/4))*4;
                float4 v={0,0,0,0};
                if(base+q<k&&bc+col<n)
                    v=*reinterpret_cast<const float4*>(b+(base+q)*n+bc+col);
                *reinterpret_cast<float4*>(sb[q]+col)=v;
            }
        }else{
            #pragma unroll
            for(int i=threadIdx.x;i<BM*BK;i+=NT){
                int r=i/BK,q=i%BK;
                sa[q][r]=br+r<m&&base+q<k?a[(br+r)*k+base+q]:0.f;
            }
            #pragma unroll
            for(int i=threadIdx.x;i<BK*BN;i+=NT){
                int q=i/BN,col=i%BN;
                sb[q][col]=base+q<k&&bc+col<n?b[(base+q)*n+bc+col]:0.f;
            }
        }
        __syncthreads();
        #pragma unroll
        for(int q=0;q<BK;++q){
            float av[TM],bv[TN];
            #pragma unroll
            for(int i=0;i<TM;++i)av[i]=sa[q][ty*TM+i];
            #pragma unroll
            for(int j=0;j<TN;++j)bv[j]=sb[q][tx+j*NX];
            #pragma unroll
            for(int i=0;i<TM;++i)
                #pragma unroll
                for(int j=0;j<TN;++j)acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<TM;++i)
        #pragma unroll
        for(int j=0;j<TN;++j){
            int r=br+ty*TM+i,col=bc+tx+j*NX;
            if(r<m&&col<n)c[r*n+col]=acc[i][j];
        }
}
template<int BM,int BN,int BK,int TM,int TN>
void launch_vector_tile(const float*a,const float*b,float*c,int m,int k,int n){
    constexpr int NT=(BM/TM)*(BN/TN);
    dim3 grid((n+BN-1)/BN,(m+BM-1)/BM);
    if(k%4==0&&n%4==0&&aligned16(a)&&aligned16(b))
        kernel_matmul_vector<BM,BN,BK,TM,TN,true><<<grid,NT,0,execution_stream()>>>(a,b,c,m,k,n);
    else kernel_matmul_vector<BM,BN,BK,TM,TN,false><<<grid,NT,0,execution_stream()>>>(a,b,c,m,k,n);
}
template<int PARTS>
__global__ void kernel_gemv_partition(const float* __restrict__ a,const float* __restrict__ b,
                                     float* __restrict__ c,int k,int n){
    __shared__ float sums[PARTS][32];
    int lane=threadIdx.x,part=threadIdx.y,col=blockIdx.x*32+lane;
    float sum=0;
    if(col<n){
        #pragma unroll 4
        for(int q=part;q<k;q+=PARTS)sum=fmaf(a[q],b[q*n+col],sum);
    }
    sums[part][lane]=sum;__syncthreads();
    if(part==0&&col<n){
        sum=0;
        #pragma unroll
        for(int q=0;q<PARTS;++q)sum+=sums[q][lane];
        c[col]=sum;
    }
}
const char* matmul_config_name(int config){
    static const char* names[]={"Vec32x32_K32","Vec32x64_K32","Vec32x64_2x4","Vec64x64_K32",
        "Vec64x64_K64","Vec16x32_K32","Vec16x64_K32","Vec32x32_K64","Gemv4","Gemv16","Gemv32"};
    if(config<0||config>=MATMUL_CONFIG_COUNT)throw std::invalid_argument("Invalid MatMul configuration.");
    return names[config];
}
bool matmul_config_supported(int config,int m){
    return config>=0&&config<MATMUL_CONFIG_COUNT&&(config<8||m==1);
}
void launch_matmul_config(int config,const float*a,const float*b,float*c,int m,int k,int n){
    validate_matmul(a,b,c,m,k,n);
    if(!matmul_config_supported(config,m))throw std::invalid_argument("Configuration not supported for this shape.");
    if(!m)return;
    switch(config){
    case 0:launch_vector_tile<32,32,32,2,2>(a,b,c,m,k,n);break;
    case 1:launch_vector_tile<32,64,32,2,2>(a,b,c,m,k,n);break;
    case 2:launch_vector_tile<32,64,32,2,4>(a,b,c,m,k,n);break;
    case 3:launch_vector_tile<64,64,32,4,4>(a,b,c,m,k,n);break;
    case 4:launch_vector_tile<64,64,64,4,4>(a,b,c,m,k,n);break;
    case 5:launch_vector_tile<16,32,32,1,2>(a,b,c,m,k,n);break;
    case 6:launch_vector_tile<16,64,32,1,4>(a,b,c,m,k,n);break;
    case 7:launch_vector_tile<32,32,64,2,2>(a,b,c,m,k,n);break;
    case 8:kernel_gemv_partition<4><<<(n+31)/32,dim3(32,4),0,execution_stream()>>>(a,b,c,k,n);break;
    case 9:kernel_gemv_partition<16><<<(n+31)/32,dim3(32,16),0,execution_stream()>>>(a,b,c,k,n);break;
    case 10:kernel_gemv_partition<32><<<(n+31)/32,dim3(32,32),0,execution_stream()>>>(a,b,c,k,n);break;
    }
    CUDA_CHECK(cudaGetLastError());
}
