# Validation record

Verified on 2026-10-07 on NVIDIA GeForce RTX 5050 Laptop GPU.

The Windows build and run_all.bat completed with exit code 0. All five CTest suites passed, including five analyzer regression cases. The original cuda_kernel_library executable also completed successfully.

Compute Sanitizer memcheck: 0 errors. Racecheck: 0 errors, 0 warnings. The edge-case suite includes odd widths, offset buffers, constants, extreme finite Softmax values, tuned dispatch, and explicit rejection of NaN/Inf comparisons.

The final benchmark contains 104 aggregate records and 520 per-trial records, across 14 reference shapes, with five trials and 50 samples per trial. All candidates passed the target benchmark's output checks before timing.

| Operator | Shape | Best custom | Vendor | Median speedup | Worst trial speedup |
|---|---|---|---|---:|---:|
| LayerNorm | 128x768 | Dispatcher | cuDNN | 1.253x | 1.197x |
| MatMul | 128x64x128 | Tiled32(V2) | cuBLAS | 1.579x | 1.442x |
| Softmax | 12288x1024 | Shape | cuDNN | 1.443x | 1.420x |

These are selected-shape wins under FP32 graph replay with cached buffers, not end-to-end model timings or a guarantee across all shapes. Losses and variable advantages remain in summary.md. cuBLAS uses pedantic FP32 math; cuDNN 9.17.1 supplies the normalization baselines.

Final Nsight Compute reports and readable details are under profiles/. LayerNorm registers fell from 80 to 40 per thread; achieved occupancy rose from 13.31% to 48.61% in the initial/final profiles. The large Softmax profile reports 93.52% DRAM throughput. Profile-instrumented CSVs are excluded from speedup evidence.

validation.json records the aggregate CSV checksum and source checksums. A subsequent rebuild or benchmark run can change those files and makes this record historical until validation is repeated.
