# Paper artifacts

This directory contains the curated, reproducible artifacts for the tensor-core
Fast Walsh-Hadamard Transform paper. It deliberately excludes the separate
`fused_ht` experiments.

The live research tracker and profiling commands are in `PROJECT_BOARD.md`.
A durable reconstruction of the project conversation is in
`CHAT_LOG_2026-08-23.md`.

## Validated H100 result

- Date: 2026-08-23
- Repository commit: `7ad3fdbc3577d0c48c747789fc97fa106fb91010`
- Slurm job: `21044277` (`mit_preemptable`, node `node2900`)
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- PyTorch: 2.9.1+cu128 (CUDA runtime 12.8)
- Extension compiler: CUDA 12.9.1
- Workload: 2^30 elements, 400 timed in-place transforms
- Dtypes: FP16 and BF16
- Sizes: 256 through 32768
- Raw data: `results/raw/h100_sm90_fwht_2026-08-23.json`
- Tri Dao comparison: `results/raw/h100_sm90_dao_comparison_2026-08-23.json`
- FlagGems comparison: `results/raw/h100_sm90_flaggems_comparison_2026-08-23.json`
- arthurfeeney/fwht comparison: `results/raw/h100_sm90_arthur_comparison_2026-08-23.json`
- SHA-256: `014e937347c079f60d13c5eb96b422256a5c7bb1fe2741ced98fa6296506e800`

The raw file is the normal asynchronous CUDA run. A preceding
`CUDA_LAUNCH_BLOCKING=1` diagnostic run is intentionally not included as a
paper result.

## Result summary

Our kernel is within 0.0--1.7% of the empirical AddOne read/modify/write
roofline at every tested size. It is effectively tied with HadaCore through
size 4096, then reaches 1.049x/1.092x speedup at 8192,
1.102x/1.150x at 16384, and 1.529x/1.586x at 32768 for FP16/BF16.

The defensible current claim is that the implementation operates at the
empirical memory roofline and is up to 1.59x faster than HadaCore for large
transforms on H100. The data does not support claiming a speedup over HadaCore
for every transform size.

Against Tri Dao's pinned CUDA FWHT (`v1.1.0.post2`), numerical agreement passes
at every size/dtype. Our H100 kernel is tied at size 512, 1.26--1.40x faster for
most sizes from 1024 through 16384, and 2.36x/2.15x faster at size 32768 for
FP16/BF16. Size 256 is 1.77x faster. These are single-process saturated-
throughput results and require independent repetitions before publication.

Against the pinned FlagGems v5.0.2 Triton kernel, correctness also passes for
all sizes/dtypes. Sizes 512 and 1024 are effectively tied; ours is 1.11--1.15x
faster at 2048--4096, 1.24--1.25x at 8192, 1.52--1.53x at 16384, and
2.23--2.25x at 32768.

Against pinned `arthurfeeney/fwht` plus our documented BF16 compatibility
patch, correctness passes for FP16 and BF16 through 32768. Ours is
1.81x/1.80x faster at size 256, 1.30x/1.30x at 512, effectively tied at
1024--2048, 1.11x/1.16x at 4096, effectively tied at 8192--16384, and
1.28x/1.36x at 32768 for FP16/BF16. Unmodified upstream does not compile for
BF16 with Triton 3.5.1; our patch uses FP32 `tl.dot` output and casts back to
BF16 at the original stage boundaries. This must be disclosed with the table.

Nsight Compute profiling against HadaCore supports the reduced-instruction
explanation. At sizes 256, 4096, and 32768, ours executes respectively
2.92x/2.79x, 2.25x/2.02x, and 3.47x/3.60x fewer dynamic warp instructions for
FP16/BF16. At 32768 this corresponds to a profiled-kernel speedup of
1.69x/1.81x. At 256 the kernels remain effectively tied despite the instruction
reduction, consistent with bandwidth and launch overhead dominating.

A preliminary H100 H16 application prototype exactly matches the dense RHT
reference for FP16 and BF16. Plain tiled RHT is within 2.3% of the dense H16
matmul latency. A fused RHT + per-16 amax/scale + native E4M3 output kernel is
6.01x/5.98x faster than an eager separate-operation pipeline for FP16/BF16 at
67.1M elements. This is a fusion ablation, not a Transformer Engine comparison;
it must not be described as native NVFP4 because H100 lacks that hardware path.

A preliminary matched Transformer Engine 2.18 sweep uses PyTorch 2.8.0+cu129 on
this H100. For a 4096x8192x8192 GEMM, delayed-scaling FP8 is 1.32x faster than
BF16 in forward and 1.66x faster for the measured forward+backward step. FP8
block scaling reaches 1.17x and 1.34x respectively, trailing delayed scaling.
On Llama-style 4096-to-11008 and 11008-to-4096 projections, delayed scaling's
forward+backward speedup is 1.47x and 1.41x; block scaling reaches 1.26x for
both. At the smaller 4096-cubed forward shape, all FP8 recipes trail BF16,
showing that quantization overhead can dominate. These are preliminary
single-process microbenchmarks, not an end-to-end model result.

See `results/h100_sm90_summary.csv` for the derived table and `baselines/` for
the external-baseline registry.

## Limitations before submission

- Repeat the sweep in independent processes and report confidence intervals.
- Randomize method order.
- Validate against an independent FP32 reference, not only HadaCore.
- Capture clocks, temperature, power, driver, compiler flags, and git status.
- Add latency/batch-size sweeps in addition to saturated throughput.
- Reproduce on A100, L40S, and H200.
- Benchmark pinned Tri Dao CUDA and FlagGems Triton baselines on identical inputs.
