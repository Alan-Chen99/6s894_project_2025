# Dynamic RHT weight profile on H100 NVL

Captured on 2026-08-23 with Nsight Systems 2025.1.3 after five warmup steps.
The matched shape is 4096 tokens, hidden size 4096, and SwiGLU intermediate
size 11008. AdamW kernels are excluded.

| Path | Projected GPU interval | GPU operations | GEMMs | Weight-quantization kernels |
|---|---:|---:|---:|---:|
| BF16 working dynamic weight | 6.375 ms | 13 | 6 | 2 |
| Direct 2D block-FP8 dynamic weight | 6.106 ms | 11 | 6 | 0 |

The directly quantized path reduces the projected GPU interval by 4.2% and
removes the two TE square-block weight-quantization operations. The six GEMMs
take 4.360 ms versus 4.393 ms, which is normal run variation. The fused path's
two activation and two weight RHT/FP8 writers total 0.733 ms. The BF16-working
path uses two activation writers totaling 0.214 ms, two RHT materialization
kernels totaling 0.417 ms, and two subsequent TE weight-quantization kernels
totaling 0.168 ms. Both paths use the same three
inverse-RHT kernels (about 0.455 ms) and fused inverse-RHT+dSwiGLU kernel
(about 0.219 ms).

The event benchmark independently interleaves the two dynamic paths and finds
7.732 ms for direct 2D FP8 versus 7.814 ms for BF16 working weights, a 1.011x
speedup. Against separately interleaved static-rotation controls, either
dynamic path still adds approximately 18--19% step time. Therefore the fused
weight writer successfully eliminates redundant TE quantization, while the
remaining target is the mandatory per-step transform and master-gradient map.

Raw reports:

- `h100_nvl_dynamic_rht_bf16_working.nsys-rep`
- `h100_nvl_dynamic_rht_primary_fp8.nsys-rep`
