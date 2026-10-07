#include "matmul.cuh"
#include "layernorm.cuh"
#include "softmax.cuh"
#include "shape_kernels.cuh"
#include "benchmark.cuh"
#include "vendor_baselines.h"
#include "kernel_dispatcher.cuh"
#include "validation.cuh"
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static FILE* trial_csv=nullptr;
struct Entry {std::string name;std::function<void()> launch;};
static std::vector<float> random_values(size_t n,unsigned seed,float scale=1.f){
    std::mt19937 rng(seed);std::uniform_real_distribution<float>d(-scale,scale);
    std::vector<float> v(n);for(float&x:v)x=d(rng);return v;
}
struct Buffer{
    float* p;explicit Buffer(size_t n):p(device_alloc<float>(n)){}
    explicit Buffer(const std::vector<float>&v):p(device_alloc_and_copy(v.data(),v.size())){}
    ~Buffer(){device_free(p);}Buffer(const Buffer&)=delete;
};
static void measure(FILE*csv,const char*op,const std::string&shape,std::vector<Entry>&entries,
                    float*output,const std::vector<float>&reference,double flops,int trials,int samples){
    std::vector<float> host(reference.size());std::vector<std::vector<BenchmarkResult>> results(entries.size());
    for(auto&e:entries){
        CUDA_CHECK(cudaMemset(output,0xff,host.size()*sizeof(float)));
        e.launch();CUDA_CHECK(cudaDeviceSynchronize());device_to_host(host.data(),output,host.size());
        if(!check_correctness(reference.data(),host.data(),host.size(),1e-4f,e.name.c_str(),1e-4f).passed)
            throw std::runtime_error(std::string(op)+" "+shape+" failed: "+e.name);
        if(std::string(op)=="Softmax"){
            size_t cols=std::stoul(shape.substr(shape.find('x')+1));
            for(size_t i=0;i<host.size();i+=cols){double sum=0;for(size_t j=0;j<cols;++j){
                if(host[i+j]<0)throw std::runtime_error("Negative Softmax probability");sum+=host[i+j];}
                if(std::abs(sum-1.)>1e-4)throw std::runtime_error("Softmax normalization failed");}
        }
    }
    BenchmarkConfig cfg;cfg.verbose=false;cfg.bench_iters=samples;
    // Rotate order between independent trials, avoiding a fixed thermal/order advantage.
    for(int t=0;t<trials;++t)for(size_t j=0;j<entries.size();++j){
        size_t i=(j+t)%entries.size();
        auto sample=run_benchmark(entries[i].name,entries[i].launch,cfg,flops);
        results[i].push_back(sample);
        fprintf(trial_csv,"%d,",t+1);write_csv_row(trial_csv,op,shape.c_str(),sample);
    }
    std::vector<BenchmarkResult> aggregate;
    for(size_t i=0;i<entries.size();++i){
        std::vector<float> med;for(auto&r:results[i])med.push_back(r.med_ms);
        std::sort(med.begin(),med.end());BenchmarkResult r=results[i][0];
        r.med_ms=med[med.size()/2];r.avg_ms=0;r.min_ms=results[i][0].min_ms;r.max_ms=0;
        for(auto&s:results[i]){r.avg_ms+=s.avg_ms/trials;r.min_ms=std::min(r.min_ms,s.min_ms);r.max_ms=std::max(r.max_ms,s.max_ms);}
        if(flops>0)r.tflops=(float)(flops/(r.avg_ms*1e9));aggregate.push_back(r);
        printf("  %-20s median=%9.6f ms avg=%9.6f ms\n",r.name.c_str(),r.med_ms,r.avg_ms);
    }
    auto vendor=std::find_if(aggregate.begin(),aggregate.end(),[](auto&r){return r.name=="cuBLAS"||r.name=="cuDNN";});
    auto best=std::min_element(aggregate.begin(),vendor,[](auto&a,auto&b){return a.med_ms<b.med_ms;});
    for(size_t i=0;i<aggregate.size();++i){
        auto&r=aggregate[i];float ratio=0,minimum=0;
        if(vendor!=aggregate.end()){
            size_t vi=vendor-aggregate.begin();ratio=vendor->med_ms/r.med_ms;minimum=INFINITY;
            for(int t=0;t<trials;++t)minimum=std::min(minimum,results[vi][t].med_ms/results[i][t].med_ms);
        }
        fprintf(csv,"%s,%s,%s,%.8f,%.8f,%.8f,%.5f,%.5f,%.5f,%.5f,%d\n",op,shape.c_str(),r.name.c_str(),
                r.avg_ms,r.min_ms,r.med_ms,r.tflops,aggregate[0].avg_ms/r.avg_ms,ratio,minimum,trials);
    }
    fflush(csv);
    if(vendor!=aggregate.end()){
        float ratio=vendor->med_ms/best->med_ms,minimum=INFINITY;
        size_t vi=vendor-aggregate.begin(),bi=best-aggregate.begin();
        for(int t=0;t<trials;++t)minimum=std::min(minimum,results[vi][t].med_ms/results[bi][t].med_ms);
        printf("  Best custom %s vs vendor: %.3fx (%s)\n",best->name.c_str(),ratio,
               trials>=3&&minimum>=1.05f?"at least 5% faster in every trial":ratio>1.f?"small advantage; repeat to confirm":"vendor is faster");
    }else printf("  Vendor comparison missing; no vendor-win claim is possible.\n");
}
int main(int argc,char**argv){
    try{
        std::filesystem::path csvpath="results/target_results.csv";int trials=5,samples=50;bool quick=false;std::vector<std::vector<int>> custom_mm,custom_ln,custom_sm;bool custom=false;
        for(int i=1;i<argc;++i){
            std::string a=argv[i];
            if(a=="--csv"&&i+1<argc)csvpath=argv[++i];
            else if(a=="--trials"&&i+1<argc)trials=std::stoi(argv[++i]);
            else if(a=="--samples"&&i+1<argc)samples=std::stoi(argv[++i]);
            else if(a=="--quick")quick=true;
            else if(a=="--matmul"&&i+3<argc){int m=std::stoi(argv[++i]),k=std::stoi(argv[++i]),n=std::stoi(argv[++i]);validate_shape(m,k);validate_shape(k,n);validate_shape(m,n);if(!m)throw std::invalid_argument("Benchmark rows must be positive.");custom_mm.push_back({m,k,n});custom=true;}
            else if((a=="--layernorm"||a=="--softmax")&&i+2<argc){int r=std::stoi(argv[++i]),c=std::stoi(argv[++i]);validate_shape(r,c);if(!r)throw std::invalid_argument("Benchmark rows must be positive.");(a=="--layernorm"?custom_ln:custom_sm).push_back({r,c});custom=true;}
            else if(a=="--help"){printf("benchmark_targets [--csv PATH] [--trials N] [--samples N] [--quick] [--matmul M K N] [--layernorm ROWS COLS] [--softmax ROWS COLS]\n");return 0;}
            else throw std::invalid_argument("Unknown or incomplete argument: "+a);
        }
        if(trials<1||samples<1)throw std::invalid_argument("Trials/samples must be positive.");
        cuda_init_check();print_gpu_info();
        printf("Reference workload: GPT-2 dimensions (hidden=768, FFN=3072, heads=12).\n"
               "FP32, CUDA graph replay, allocations/transfers/plan setup excluded; %d trials x %d samples.\n",trials,samples);
        if(!csvpath.parent_path().empty())std::filesystem::create_directories(csvpath.parent_path());
        FILE*csv=fopen(csvpath.string().c_str(),"w");if(!csv)throw std::runtime_error("Cannot create CSV.");
        fprintf(csv,"operator,shape,kernel,avg_ms,min_ms,med_ms,tflops,speedup_vs_naive,vendor_speedup_median,vendor_speedup_min_trial,trial_count\n");
        auto trial_path=csvpath;trial_path.replace_filename(csvpath.stem().string()+".trials.csv");
        trial_csv=fopen(trial_path.string().c_str(),"w");if(!trial_csv)throw std::runtime_error("Cannot create trial CSV.");
        fprintf(trial_csv,"trial,");write_csv_header(trial_csv);
        cublasHandle_t blas;CUBLAS_CHECK(cublasCreate(&blas));
        CUBLAS_CHECK(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
        VendorBaselines vendor;printf("Vendor normalization baseline: %s\n",vendor.version());
        auto meta_path=csvpath;meta_path.replace_extension(".metadata.json");
        cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));int runtime=0,driver=0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));CUDA_CHECK(cudaDriverGetVersion(&driver));
        std::ofstream meta(meta_path);
        meta<<"{\n  \"gpu\": \""<<prop.name<<"\",\n  \"compute_capability\": \""<<prop.major<<"."<<prop.minor
            <<"\",\n  \"cuda_runtime\": "<<runtime<<",\n  \"cuda_driver_api\": "<<driver
            <<",\n  \"vendor_norm\": \""<<vendor.version()<<"\",\n  \"precision\": \"FP32 (cuBLAS pedantic math)\",\n"
            <<"  \"protocol\": \"CUDA graph replay; cached-buffer steady-state GPU time; setup/transfers excluded\",\n"
            <<"  \"trials\": "<<trials<<",\n  \"samples_per_trial\": "<<samples<<",\n  \"warmup_iterations\": 20,\n"
            <<"  \"max_launches_per_sample\": 64,\n  \"input_seeds\": [1,2,11,12,13]\n}\n";
        bool incomplete=!vendor.available();
        std::vector<std::vector<int>> shapes={{1,768,768},{16,768,768},{128,768,768},{128,768,3072},{128,3072,768},{128,64,128}};
        if(quick)shapes={{16,768,768},{128,64,128}};
        if(custom)shapes=custom_mm;
        for(auto&s:shapes){
            int m=s[0],k=s[1],n=s[2];std::string shape=std::to_string(m)+"x"+std::to_string(k)+"x"+std::to_string(n);
            printf("\nMatMul %s\n",shape.c_str());
            auto a=random_values((size_t)m*k,1,0.5f),b=random_values((size_t)k*n,2,0.5f);
            Buffer da(a),db(b),dc((size_t)m*n);
            std::vector<float>reference((size_t)m*n);
            launch_matmul_cublas(blas,da.p,db.p,dc.p,m,k,n);device_to_host(reference.data(),dc.p,reference.size());
            // Independent CPU verification when inexpensive; all shapes compare custom output to vendor output.
            if((int64_t)m*k*n<=20000000){std::vector<float>cpu(reference.size());cpu_matmul(a.data(),b.data(),cpu.data(),m,k,n);
                if(!check_correctness(cpu.data(),reference.data(),cpu.size(),1e-3f,"cuBLAS-CPU",1e-4f).passed)throw std::runtime_error("cuBLAS reference failed");}
            cuda_kernels::autotune_matmul(m,k,n);
            std::vector<Entry>e={
                {"Naive(V1)",[&]{launch_matmul_v1(da.p,db.p,dc.p,m,k,n);}},
                {"Tiled16(V2)",[&]{launch_matmul_v2<16>(da.p,db.p,dc.p,m,k,n);}},
                {"Tiled32(V2)",[&]{launch_matmul_v2<32>(da.p,db.p,dc.p,m,k,n);}},
                {"RegBlock(V3)",[&]{launch_matmul_v3(da.p,db.p,dc.p,m,k,n);}},
                {"Unrolled32(V4)",[&]{launch_matmul_v4(da.p,db.p,dc.p,m,k,n);}},
                {"Shape",[&]{launch_matmul_shape(da.p,db.p,dc.p,m,k,n);}},
                {"Dispatcher",[&]{cuda_kernels::matmul(da.p,db.p,dc.p,m,k,n);}},
                {"cuBLAS",[&]{launch_matmul_cublas(blas,da.p,db.p,dc.p,m,k,n);}}};
            for(int config=0;config<MATMUL_CONFIG_COUNT;++config){
                if(matmul_config_supported(config,m))
                    e.insert(e.end()-1,{matmul_config_name(config),[&,config]{launch_matmul_config(config,da.p,db.p,dc.p,m,k,n);}});
            }
            measure(csv,"MatMul",shape,e,dc.p,reference,2.*m*k*n,trials,samples);
        }
        for(const std::string op:{"LayerNorm","Softmax"}){
            std::vector<std::vector<int>> rs=op=="LayerNorm"?std::vector<std::vector<int>>{{1,768},{16,768},{128,768},{128,3072}}:
                                                                      std::vector<std::vector<int>>{{12,128},{1536,128},{3072,256},{12288,1024}};
            if(quick)rs=op=="LayerNorm"?std::vector<std::vector<int>>{{128,768}}:std::vector<std::vector<int>>{{1536,128}};
            if(custom)rs=op=="LayerNorm"?custom_ln:custom_sm;
            for(auto&s:rs){
                int r=s[0],c=s[1];std::string shape=std::to_string(r)+"x"+std::to_string(c);
                printf("\n%s %s\n",op.c_str(),shape.c_str());
                auto x=random_values((size_t)r*c,11,3.f),g=random_values(c,12),b=random_values(c,13);
                Buffer dx(x),dg(g),db(b),dy((size_t)r*c);std::vector<float>ref((size_t)r*c);
                std::vector<Entry>entries;
                if(op=="LayerNorm"){
                    cpu_layernorm(x.data(),g.data(),b.data(),ref.data(),r,c);
                    entries={{"LayerNorm-V1",[&]{launch_layernorm_v1(dx.p,dg.p,db.p,dy.p,r,c);}},
                             {"LayerNorm-V2",[&]{launch_layernorm_v2(dx.p,dg.p,db.p,dy.p,r,c);}},
                             {"LayerNorm-V3",[&]{launch_layernorm_v3(dx.p,dg.p,db.p,dy.p,r,c);}},
                             {"LayerNorm-V4",[&]{launch_layernorm_v4(dx.p,dg.p,db.p,dy.p,r,c);}},
                             {"Shape",[&]{launch_layernorm_shape(dx.p,dg.p,db.p,dy.p,r,c);}}};
                    cuda_kernels::autotune_layernorm(r,c);
                    entries.push_back({"Dispatcher",[&]{cuda_kernels::layernorm(dx.p,dg.p,db.p,dy.p,r,c);}});
                    if(vendor.available()){
                        try{vendor.prepare_layernorm(r,c);entries.push_back({"cuDNN",[&]{vendor.layernorm(dx.p,dg.p,db.p,dy.p);}});}
                        catch(const std::exception&e){fprintf(stderr,"[MISSING BASELINE] %s\n",e.what());incomplete=true;}
                    }
                }else{
                    cpu_softmax(x.data(),ref.data(),r,c);
                    entries={{"Softmax-V1",[&]{launch_softmax_v1(dx.p,dy.p,r,c);}},
                             {"Softmax-V2",[&]{launch_softmax_v2(dx.p,dy.p,r,c);}},
                             {"Softmax-V3",[&]{launch_softmax_v3(dx.p,dy.p,r,c);}},
                             {"Softmax-V4",[&]{launch_softmax_v4(dx.p,dy.p,r,c);}},
                             {"Shape",[&]{launch_softmax_shape(dx.p,dy.p,r,c);}}};
                    cuda_kernels::autotune_softmax(r,c);
                    entries.push_back({"Dispatcher",[&]{cuda_kernels::softmax(dx.p,dy.p,r,c);}});
                    if(vendor.available()){vendor.prepare_softmax(r,c);entries.push_back({"cuDNN",[&]{vendor.softmax(dx.p,dy.p);}});}
                }
                measure(csv,op.c_str(),shape,entries,dy.p,ref,0,trials,samples);
            }
        }
        CUBLAS_CHECK(cublasDestroy(blas));fclose(csv);fclose(trial_csv);
        printf("\nCSV: %s\n",csvpath.string().c_str());
        if(incomplete){fprintf(stderr,"Vendor comparison is incomplete; benchmark status=2.\n");return 2;}
        return 0;
    }catch(const std::exception&e){fprintf(stderr,"[ERROR] %s\n",e.what());return 1;}
}
