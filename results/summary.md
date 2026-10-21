# Benchmark Summary

Latency: med_ms. Ratios above 1 mean custom is faster. V1 means the actual V1 row. A vendor win requires at least 5% improvement in every measured trial.

| Operator | Shape | V1 ms | Best custom | Custom ms | Vendor | Vendor ms | vs V1 | vs vendor | Evidence |
|---|---|---:|---|---:|---|---:|---:|---:|---|
| LayerNorm | 128x3072 | 0.011991 | Shape | 0.005976 | cuDNN | 0.005904 | 2.007x | 0.988x | vendor faster |
| LayerNorm | 128x768 | 0.005093 | Shape | 0.002627 | cuDNN | 0.003312 | 1.938x | 1.261x | consistency threshold not met |
| LayerNorm | 16x768 | 0.002758 | Dispatcher | 0.001906 | cuDNN | 0.001422 | 1.447x | 0.746x | vendor faster |
| LayerNorm | 1x768 | 0.002754 | Shape | 0.001945 | cuDNN | 0.001333 | 1.415x | 0.685x | vendor faster |
| MatMul | 128x3072x768 | 0.853440 | Vec64x64_K64 | 0.361856 | cuBLAS | 0.099221 | 2.359x | 0.274x | vendor faster |
| MatMul | 128x64x128 | 0.004804 | Vec32x32_K32 | 0.003086 | cuBLAS | 0.007466 | 1.557x | 2.419x | consistency threshold not met |
| MatMul | 128x768x3072 | 0.858816 | Vec64x64_K32 | 0.211480 | cuBLAS | 0.111776 | 4.061x | 0.529x | vendor faster |
| MatMul | 128x768x768 | 0.208520 | Vec64x64_K64 | 0.092949 | cuBLAS | 0.042755 | 2.243x | 0.460x | vendor faster |
| MatMul | 16x768x768 | 0.039216 | Tiled16(V2) | 0.034773 | cuBLAS | 0.011527 | 1.128x | 0.331x | vendor faster |
| MatMul | 1x768x768 | 0.016336 | Gemv16 | 0.003885 | cuBLAS | 0.004705 | 4.205x | 1.211x | consistency threshold not met |
| Softmax | 12288x1024 | 0.496224 | Softmax-V4 | 0.382624 | cuDNN | 0.562400 | 1.297x | 1.470x | >=5% faster in every trial |
| Softmax | 12x128 | 0.002012 | Shape | 0.001620 | cuDNN | 0.001708 | 1.242x | 1.054x | consistency threshold not met |
| Softmax | 1536x128 | 0.029947 | Shape | 0.002904 | cuDNN | 0.003315 | 10.314x | 1.142x | consistency threshold not met |
| Softmax | 3072x256 | 0.061555 | Dispatcher | 0.006650 | cuDNN | 0.006685 | 9.257x | 1.005x | consistency threshold not met |
