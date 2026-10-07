# Benchmark Summary

Latency: med_ms. Ratios above 1 mean custom is faster. V1 means the actual V1 row. A vendor win requires at least 5% improvement in every measured trial.

| Operator | Shape | V1 ms | Best custom | Custom ms | Vendor | Vendor ms | vs V1 | vs vendor | Evidence |
|---|---|---:|---|---:|---|---:|---:|---:|---|
| LayerNorm | 128x3072 | 0.012345 | Shape | 0.005846 | cuDNN | 0.006135 | 2.112x | 1.049x | consistency threshold not met |
| LayerNorm | 128x768 | 0.004923 | Dispatcher | 0.002307 | cuDNN | 0.003119 | 2.134x | 1.352x | >=5% faster in every trial |
| LayerNorm | 16x768 | 0.003150 | Shape | 0.002337 | cuDNN | 0.001815 | 1.348x | 0.776x | vendor faster |
| LayerNorm | 1x768 | 0.003767 | Dispatcher | 0.002080 | cuDNN | 0.001361 | 1.811x | 0.654x | vendor faster |
| MatMul | 128x3072x768 | 0.938224 | Shape | 0.528944 | cuBLAS | 0.100816 | 1.774x | 0.191x | vendor faster |
| MatMul | 128x64x128 | 0.004768 | Tiled32(V2) | 0.004652 | cuBLAS | 0.007875 | 1.025x | 1.693x | >=5% faster in every trial |
| MatMul | 128x768x3072 | 0.885072 | Shape | 0.259984 | cuBLAS | 0.118821 | 3.404x | 0.457x | vendor faster |
| MatMul | 128x768x768 | 0.213128 | Shape | 0.134517 | cuBLAS | 0.043258 | 1.584x | 0.322x | vendor faster |
| MatMul | 16x768x768 | 0.039397 | Dispatcher | 0.034864 | cuBLAS | 0.011333 | 1.130x | 0.325x | vendor faster |
| MatMul | 1x768x768 | 0.016245 | Dispatcher | 0.007035 | cuBLAS | 0.004706 | 2.309x | 0.669x | vendor faster |
| Softmax | 12288x1024 | 0.475632 | Shape | 0.380544 | cuDNN | 0.543888 | 1.250x | 1.429x | >=5% faster in every trial |
| Softmax | 12x128 | 0.002456 | Dispatcher | 0.001256 | cuDNN | 0.001388 | 1.956x | 1.106x | consistency threshold not met |
| Softmax | 1536x128 | 0.028986 | Dispatcher | 0.002587 | cuDNN | 0.002495 | 11.205x | 0.965x | vendor faster |
| Softmax | 3072x256 | 0.075452 | Shape | 0.006754 | cuDNN | 0.006904 | 11.172x | 1.022x | consistency threshold not met |
