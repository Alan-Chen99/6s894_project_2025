# H100 NVL TE RHT training profile

Nsight Systems 2025.1.3 captured one warmed square-4096 training step with
PyTorch 2.8.0+cu129 and Transformer Engine 2.18.0. The separate and combined
paths used the same layer, block-FP8 recipe, input, and output gradient.

| Metric | Separate RHT + TE | Combined RHT/FP8 | Change |
|---|---:|---:|---:|
| NVTX GPU-projected interval | 623.334 us | 563.397 us | -9.6% |
| Input RHT/quantization launches | 2 | 1 | -1 launch |
| Combined RHT/row+column writer | -- | 56.544 us | -- |
| Three TE GEMMs | 492.005 us | 492.067 us | effectively unchanged |
| Inverse RHT | 56.160 us | 55.841 us | effectively unchanged |

The profile supports the conclusion that backward GEMMs and inverse RHT are
not regressing. The remaining discrepancy between the GPU trace and eager
CUDA-event medians at square-4096 is host dispatch/allocation variability.
The binary `.nsys-rep` files were kept in node-local temporary storage rather
than committed because they are large and architecture-specific.
