# H100 NVL external RHT + Transformer Engine comparison

This experiment compares the project's fused RHT--block-FP8 MLP adapter with a
functionality-matched composition of independently available components:

- unmodified Tri Dao `fast-hadamard-transform` at commit
  `e7706faf8d1c3b9f241e36860640ad1dac644ede`, rebuilt for the PyTorch 2.8.0
  ABI with SM90-only code generation;
- a separate stock PyTorch BF16 multiply for the fixed diagonal Rademacher
  signs (`0xA3F5`); and
- unmodified Transformer Engine 2.18.0 block-scaled FP8.

The H16 kernel itself is off the shelf. The sign multiply is explicitly kept
separate because the upstream API implements Hadamard rather than randomized
Hadamard. Both methods compute the same normalized transform `D H / 4`; their
BF16 outputs match bit-for-bit on the correctness input (zero max-absolute and
relative-L2 difference). The Llama-shaped workload has 4096 tokens, hidden
size 4096, intermediate size 11008, combined FC1, fused SwiGLU, and a down
projection. Corresponding input-axis weight rotations are performed once
outside timing. Timings include forward or forward+backward, but no optimizer.

## Independent-process timing

| Phase | External RHT + TE | Project fused path | Speedup | Process range |
|---|---:|---:|---:|---:|
| Forward | 4.533 ms | 1.964 ms | 2.308x | 2.306--2.309x |
| Backward only | 6.171 ms | 3.488 ms | 1.769x | 1.767--1.775x |
| Forward + backward | 10.877 ms | 5.508 ms | 1.974x | 1.969--1.981x |

The figures are medians across three independently started processes. Within
each process, 30 samples per method are measured as interleaved randomized
pairs after ten warmups.

For context, the same fused path is 0.941x in forward and 0.946x in
forward+backward versus TE block FP8 with no RHT. Therefore the application
claim is a reduction in the cost of an RHT-enabled pipeline, not a speedup over
omitting RHT.

## CUDA attribution

PyTorch's CUDA profiler attributes one external forward+backward step as
follows. Times for raw CUDA kernels are shown; enclosing autograd/framework
events are excluded to avoid double counting.

| Work | External RHT + TE | Project fused path |
|---|---:|---:|
| Tri Dao H16 | 5.119 ms / 4 launches | -- |
| PyTorch sign multiply | 0.370 ms / 4 launches | -- |
| Project RHT/FP8 writers | -- | 0.261 ms / 2 launches |
| Project inverse RHT kernels | -- | 0.284 ms / 2 launches |
| TE FP8 cast/transpose kernels | 0.489 ms / 6 launches | 0.405 ms / 4 launches |
| TE GEMMs | 4.242 ms / 6 launches | 4.251 ms / 6 launches |

The GEMM time is effectively unchanged. The gap is explained by replacing
about 5.49 ms of separate upstream H16 plus sign work, and some TE casting,
with roughly 0.54 ms of fused RHT/quantization and inverse-transform kernels.
The profiler is an attribution run, not the source of the headline timings.

Raw timing and profile JSON files are under `paper/results/raw/` with prefixes
`h100_nvl_node4508_dao_te_vs_fused_rht_`,
`h100_nvl_node4508_dao_te_profile_`, and
`h100_nvl_node4508_fused_rht_te_profile_`.
