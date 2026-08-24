# H100 NVL fused inverse-RHT AdamW profile

- Date: 2026-08-24
- Slurm node: `node4508`
- GPU: NVIDIA H100 NVL, SM90
- Shape: 4096 tokens, 4096 hidden, 11008 intermediate
- Software: PyTorch 2.8.0+cu129, Transformer Engine 2.18.0,
  Nsight Systems 2025.1.3
- Trace: `h100_nvl_node4508_fused_rht_adamw.nsys-rep`

The mapped reference writes BF16 working Wgrad through $R^T$ into a separate
FP32 master-gradient buffer and then invokes foreach PyTorch AdamW. The fused
path consumes rotated BF16 Wgrad directly and combines inverse H16, FP32 first
and second moments, decoupled weight decay, and the original-basis parameter
update. Both paths use two weight matrices.

| NVTX range | Projected GPU interval (ms) | GPU operations | Speedup |
|---|---:|---:|---:|
| Mapped full step | 9.525 | 67 | -- |
| Fused full step | 6.810 | 11 | 1.399x |
| Mapped optimizer tail | 3.903 | 58 | -- |
| Fused optimizer tail | 1.172 | 2 | 3.332x |

The two fused tail operations are one update launch per weight. Nsight
projected intervals are a single profiled execution and are separate from the
randomized event-timing medians (1.407x full step and 3.132x optimizer tail).

Capture command:

```bash
module load cuda/12.9.1
nsys profile \
  --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  --cuda-event-trace=false \
  --force-overwrite=true \
  -o paper/profiles/h100_nvl_node4508_fused_rht_adamw \
  .venv-cu129/bin/python paper/scripts/profile_fused_rht_adamw_h100.py

nsys stats --report nvtx_gpu_proj_sum --format csv \
  paper/profiles/h100_nvl_node4508_fused_rht_adamw.nsys-rep
```
