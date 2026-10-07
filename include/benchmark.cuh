#pragma once
#include "cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct BenchmarkConfig {
    int warmup_iters=20, bench_iters=100;
    bool verbose=true;
    int max_launches_per_sample=64;
    bool use_cuda_graph=true;
};
struct BenchmarkResult {
    std::string name;
    float avg_ms=0,min_ms=0,med_ms=0,max_ms=0,tflops=0;
    double gflops_total=0;
    int launches_per_sample=1;
};
inline BenchmarkResult run_benchmark(const std::string& name,std::function<void()> fn,
                                     const BenchmarkConfig& cfg=BenchmarkConfig(),double flops=0) {
    if(cfg.warmup_iters<0||cfg.bench_iters<1||cfg.max_launches_per_sample<1)
        throw std::invalid_argument("Invalid benchmark iteration count.");
    cudaStream_t stream;CUDA_CHECK(cudaStreamCreate(&stream));
    cudaStream_t previous=execution_stream();set_execution_stream(stream);
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    try {
        CudaTimer timer;
        for(int i=0;i<cfg.warmup_iters;++i)fn();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        timer.begin();fn();timer.end();
        float pilot=timer.elapsed_ms();
        int repeats=std::max(1,std::min(cfg.max_launches_per_sample,(int)std::ceil(0.3f/std::max(pilot,0.0001f))));
        if(cfg.use_cuda_graph) {
            CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for(int i=0;i<repeats;++i)fn();
            CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
            CUDA_CHECK(cudaGraphLaunch(executable,stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        std::vector<float> times;times.reserve(cfg.bench_iters);
        for(int i=0;i<cfg.bench_iters;++i) {
            timer.begin();
            if(executable)CUDA_CHECK(cudaGraphLaunch(executable,stream));
            else for(int j=0;j<repeats;++j)fn();
            timer.end();times.push_back(timer.elapsed_ms()/repeats);
        }
        std::sort(times.begin(),times.end());
        BenchmarkResult r;
        r.name=name;r.avg_ms=std::accumulate(times.begin(),times.end(),0.f)/times.size();
        r.min_ms=times.front();r.max_ms=times.back();
        size_t mid=times.size()/2;r.med_ms=times.size()%2?times[mid]:(times[mid-1]+times[mid])/2;
        r.launches_per_sample=repeats;
        if(flops>0){r.tflops=(float)(flops/(r.avg_ms*1e9));r.gflops_total=flops/1e9;}
        if(cfg.verbose)printf("  %-26s avg=%9.6f ms min=%9.6f med=%9.6f TFLOPS=%.3f (%d launches/sample)\n",
                              name.c_str(),r.avg_ms,r.min_ms,r.med_ms,r.tflops,repeats);
        if(executable)CUDA_CHECK(cudaGraphExecDestroy(executable));
        if(graph)CUDA_CHECK(cudaGraphDestroy(graph));
        set_execution_stream(previous);CUDA_CHECK(cudaStreamDestroy(stream));
        return r;
    }catch(...){
        cudaStreamCaptureStatus status;cudaStreamIsCapturing(stream,&status);
        if(status!=cudaStreamCaptureStatusNone){cudaGraph_t invalid=nullptr;cudaStreamEndCapture(stream,&invalid);if(invalid)cudaGraphDestroy(invalid);}
        if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        set_execution_stream(previous);cudaStreamDestroy(stream);throw;
    }
}
struct CorrectnessResult {bool passed;float max_abs_err,max_rel_err;double avg_abs_err;};
inline CorrectnessResult check_correctness(const float* ref,const float* gpu,size_t n,
                                           float abs_tol=1e-4f,const char* op="operator",float rel_tol=0.f) {
    if(!n||!ref||!gpu||abs_tol<0||rel_tol<0)throw std::invalid_argument("Invalid correctness check.");
    float max_abs=0,max_rel=0;double sum=0;bool passed=true;
    for(size_t i=0;i<n;++i) {
        if(!std::isfinite(ref[i])||!std::isfinite(gpu[i])) {
            passed=false;max_abs=INFINITY;max_rel=INFINITY;sum=INFINITY;continue;
        }
        float diff=std::abs(ref[i]-gpu[i]),rel=diff/std::max(std::abs(ref[i]),1e-8f);
        max_abs=std::max(max_abs,diff);max_rel=std::max(max_rel,rel);sum+=diff;
        if(diff>abs_tol+rel_tol*std::abs(ref[i]))passed=false;
    }
    printf("  Correctness %-20s %s (max_abs=%.3e avg_abs=%.3e abs_tol=%.1e rel_tol=%.1e)\n",
            op,passed?"PASS":"FAIL",max_abs,sum/n,abs_tol,rel_tol);
    return {passed,max_abs,max_rel,sum/n};
}
inline void print_speedup(const BenchmarkResult& baseline,const BenchmarkResult& optimized) {
    printf("  Speedup %s -> %s: %.3fx\n",baseline.name.c_str(),optimized.name.c_str(),baseline.avg_ms/optimized.avg_ms);
}
inline void write_csv_header(FILE* f) {
    fprintf(f,"operator,shape,kernel,avg_ms,min_ms,med_ms,tflops,speedup_vs_naive\n");
}
inline void write_csv_row(FILE* f,const char* op,const char* shape,const BenchmarkResult& r,float speedup=1.f) {
    fprintf(f,"%s,%s,%s,%.8f,%.8f,%.8f,%.5f,%.5f\n",op,shape,r.name.c_str(),
            r.avg_ms,r.min_ms,r.med_ms,r.tflops,speedup);
}
