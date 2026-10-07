# Reference target workload

The project uses a GPT-2-style workload because no production model was supplied. The published [GPT-2 configuration](https://huggingface.co/openai-community/gpt2/blob/main/config.json) specifies hidden size 768, 12 heads, and 1024 positions. The [model implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) uses a default intermediate width of 4 times hidden size (3072).

These are operator microbenchmarks with synthetic deterministic inputs. We do not download weights or claim production/model-level latency results. Sequence lengths and token counts below are selected test scenarios.

| Operator | Shape | Interpretation |
|---|---|---|
| MatMul | 1 x 768 x 768 | One-token hidden projection |
| MatMul | 16 x 768 x 768 | Short-token hidden projection |
| MatMul | 128 x 768 x 768 | Prefill hidden projection |
| MatMul | 128 x 768 x 3072 | MLP expansion |
| MatMul | 128 x 3072 x 768 | MLP contraction |
| MatMul | 128 x 64 x 128 | One attention head's score calculation; B is already laid out as K-transpose |
| LayerNorm | 1/16/128 x 768 | Hidden-feature normalization at different token counts |
| LayerNorm | 128 x 3072 | Intermediate-feature normalization stress case; GPT-2 does not place its standard LayerNorm here |
| Softmax | 12 x 128 | One attention query for 12 heads, context 128 |
| Softmax | 1536 x 128 | 12 heads x 128 query positions, context 128 |
| Softmax | 3072 x 256 | 12 heads x 256 query positions, context 256 |
| Softmax | 12288 x 1024 | 12 heads x 1024 query positions, context 1024 |

Softmax here is unmasked row-wise Softmax; causal masking, attention scaling, batching and fused attention are outside this operator benchmark. MatMul timings do not include preparing/transposing K. All operands/outputs use FP32 and the same contiguous layout for the custom/vendor comparison.

To evaluate another model, pass --matmul M K N, --layernorm ROWS HIDDEN or --softmax ROWS COLS to benchmark_targets. Document the source and meaning of the chosen shapes before presenting them as model-specific results.
