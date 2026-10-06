# Benchmark Summary

| Operator | Shape | Naive (ms) | Optimized (ms) | cuBLAS (ms) | Opt Speedup | vs cuBLAS |
|----------|-------|-----------|----------------|-------------|-------------|----------|
| LayerNorm | 128x4096 | 0.022 | 0.018 | N/A | 1.20x | N/A |
| LayerNorm | 32x1024 | 0.014 | 0.012 | N/A | 1.21x | N/A |
| LayerNorm | 32x4096 | 0.017 | 0.014 | N/A | 1.19x | N/A |
| MatMul | 1024x1024x1024 | 2.885 | 0.948 | 0.331 | 3.05x | 0.35x |
| MatMul | 2048x2048x2048 | 22.830 | 7.279 | 2.003 | 3.14x | 0.28x |
| MatMul | 512x512x512 | 1.583 | 0.160 | 0.069 | 9.91x | 0.43x |
| Softmax | 128x4096 | 0.024 | 0.015 | N/A | 1.61x | N/A |
| Softmax | 32x1024 | 0.018 | 0.013 | N/A | 1.36x | N/A |
| Softmax | 32x4096 | 0.028 | 0.015 | N/A | 1.92x | N/A |
