# Paper artifacts

This directory contains the curated, reproducible artifacts for the tensor-core
Fast Walsh-Hadamard Transform paper. It deliberately excludes the separate
`fused_ht` experiments.

The live research tracker and profiling commands are in `PROJECT_BOARD.md`.
A durable reconstruction of the project conversation is in
`CHAT_LOG_2026-08-23.md`.
Manuscript edits follow `PAPER_WRITING_STYLE.md`; `AGENTS.md` makes that guide
the default for future work under this directory.

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

An initial forward pipeline experiment applies the fixed-sign H16 RHT before
the same TE Linear. Delayed FP8 plus RHT remains 1.19x faster than plain BF16
Linear for square-8192 and 1.13x faster for the Llama up projection. It is
approximately tied for Llama down (0.98x) and slower for square-4096. Against
BF16 plus the same RHT, delayed FP8 is 1.30x, 1.27x, and 1.13x faster for
square-8192, Llama up, and Llama down. This isolates forward pipeline cost; it
is not a TE quantized-tensor adapter or a training benchmark.

We subsequently implemented a direct Hopper adapter that fuses eight H16
transforms with TE-compatible block-128 power-of-two scaling and writes a
`Float8BlockwiseQTensor` without requantization. Its scales exactly match TE
and its reconstruction MSE is identical to separate RHT plus TE quantization.
The fused quantizer is 1.27--1.34x faster, producing a 1.02--1.10x full forward
pipeline speedup over the separate implementation. For square-8192 and the two
Llama projections, the fused RHT pipeline is only 4.3%, 6.5%, and 2.4% slower
than TE block-FP8 Linear without any rotation in the forward-only experiment.

The training adapter emits TE's independently scaled columnwise view for Wgrad
and applies the exact transpose RHT to Dgrad. A single 128x128 Triton tile now
computes H16 once and writes both TE layouts, matching the two-writer output
byte-for-byte and scale-for-scale. Two-view preprocessing is 1.20--1.37x faster
than separate RHT plus TE quantization; forward ranges from 1.01--1.12x and the
full training step from 0.99--1.04x. Nsight measures 9.6% less GPU interval at
square-4096 with unchanged GEMM and inverse-RHT costs.

The Llama-shaped MLP test uses combined FC1, TE fused SwiGLU, a down projection,
and matching input-axis rotations for both weights. The final Hopper hybrid
keeps TE SwiGLU in forward and fuses inverse RHT with dSwiGLU in backward.
Across three processes it is 1.054x/1.020x/1.027x faster than our separate
Triton RHT plus TE in forward/backward/forward-plus-backward. This is an
internal fusion ablation, not an external-baseline result. FP32 paired outputs and gradients agree
within 5.7e-7 relative L2; BF16 differs by
about 0.6%. Paired TE block FP8 has 6.56--6.82% error against BF16, marginally
below plain block FP8 for output and all three gradients. Those static-weight
timings remain useful kernel/integration measurements; H100 NVL absolute
timings stay separate from H100 HBM3.

The functionality-matched external comparison replaces that separate project
RHT with the unmodified, pinned Tri Dao H16 kernel, retains an explicit stock
PyTorch BF16 multiply for the diagonal Rademacher signs, and uses unmodified TE
2.18 block FP8. The upstream and project transforms match bit-for-bit in BF16.
Across three independent processes at the same 4096-token Llama shape, our
fused path is 2.308x faster in forward (4.533 to 1.964 ms), 1.769x in backward
(6.171 to 3.488 ms), and 1.974x for forward plus backward (10.877 to 5.508 ms).
The respective ranges are 2.306--2.309x, 1.767--1.775x, and 1.969--1.981x.
CUDA attribution assigns 5.119 ms to four upstream H16 launches and 0.370 ms
to four sign-multiply launches; TE GEMM time is effectively identical. Against
TE block FP8 with no RHT, however, the fused path is 0.941x in forward and
0.946x for forward plus backward. Thus 1.974x is the RHT-enabled external-
composition claim, not a speedup over omitting RHT. See
`results/h100_nvl_dao_te_external_rht_summary.csv` and
`profiles/h100_nvl_dao_te_external_rht_profile_summary.md`.

The dynamic training path now keeps FP32 AdamW masters and moments in the
original basis, writes transformed weights directly into TE's 128x128 2D
block-FP8 storage, and maps BF16 Wgrad through the exact transpose before the
optimizer. Its weight bytes/scales match TE's reference exactly, the mapped
gradient differs from dense FP32 by 1.85e-9 relative L2, and the AdamW update
matches exactly. In a controlled 120-step teacher--student SwiGLU run, dynamic
RHT block-FP8 ends at 0.001847 MSE versus 0.001907 for ordinary block-FP8 and
0.000331 for BF16. RHT therefore does not worsen the observed FP8 convergence
floor, but neither FP8 path matches BF16 quality on this task. At the
4096-to-11008 shape, the directly quantized path is 1.011x faster than the
interleaved BF16-working dynamic control (7.732 versus 7.814 ms). Nsight shows
11 versus 13 GPU operations and a 4.2% shorter projected interval. Dynamic
weight handling still costs about 18--19% versus paired static-weight steps,
making materialization and gradient mapping the next optimization target. See
`plots/h100_dynamic_rht_convergence.svg` and the companion cost plots.

We now fuse the inverse H16 transform directly into the original-basis FP32
AdamW update, eliminating the materialized master-gradient buffer and its
separate optimizer launches. A strict independently owned two-path check gives
zero output/loss/working-gradient difference; after one update the FP32 masters
agree within 4.8e-8 relative L2 and moments within 2.4e-7. In the exact
one-to-one pipeline, forward is 1.000x and backward 0.997x because those phases
are identical. Across three processes at the 4096-token,
4096-to-22016-to-4096 SwiGLU shape, the fused path takes 8.356 ms versus 11.233
ms for transpose-map plus foreach PyTorch AdamW (1.370x median full-step
speedup, range 1.344--1.399x); the optimizer tail alone improves from 4.224 to
1.338 ms (3.132x). In the matched
120-step controlled task it improves median step time from 1.221 to 1.029 ms
(1.186x), while final MSE differs by 1.6% (0.001817 fused versus 0.001847
mapped). These are single-process H100 NVL results and still require independent
repetitions and a language-model time-to-quality experiment. See
`plots/h100_fused_rht_adamw_performance.svg` and
`plots/h100_fused_rht_adamw_convergence.svg`.
Nsight independently projects 9.525 versus 6.810 ms and 67 versus 11 GPU
operations for one full step; its isolated optimizer range drops from 3.903
ms/58 operations to 1.172 ms/2 operations. The detailed trace summary is in
`profiles/h100_nvl_fused_rht_adamw_profile_summary.md`.

The separate system-level baseline uses ordinary TE block-FP8 with the same
original-basis FP32 masters and foreach AdamW. Across three independent
processes at the same model shape, ours is 0.842x in forward, 0.952x in
backward, and 0.898x for forward+backward: the RHT integration still adds
compute before the optimizer. Once AdamW is included, the complete step is
1.312x faster (range 1.310--1.318x), with median latency 7.859 versus 10.295
ms. Output relative L2 against an independent BF16 computation is 0.0661 for
ours and 0.0662 for plain TE, so the end-to-end gain does not come with a
larger error in this numerical check. See
`plots/h100_te_vs_fused_rht_adamw.svg` and
`results/h100_nvl_te_vs_fused_rht_adamw_summary.csv`.
This comparison measures net cost versus omitting RHT; unlike the optimizer
ablation above, it is intentionally not an operation-for-operation pipeline.

The corrected convergence artifact is
`results/raw/h100_nvl_node4508_te_rht_convergence_independent_masters_cu129_2026-08-24.json`.
It supersedes the 2026-08-23 multi-path convergence JSONs, whose already-FP32
contiguous initializer was inadvertently shared between optimizer Parameters.
The bridge now unconditionally clones master storage, and the strict check
asserts that the two paths have different storage addresses.

A paired-weight shape sweep finds the best forward result at the
3584-to-18944 high-expansion shape with 8192 tokens: 1.073x forward and 1.032x
full training versus separate RHT plus TE. At 1024 tokens the same shape falls
to 1.020x/1.008x, while an 8192-to-28672 case at 2048 tokens reaches
1.025x/1.012x. High expansion plus enough tokens to saturate the fused writer
is more favorable than model width by itself.

See `results/h100_sm90_summary.csv` for the derived table and `baselines/` for
the external-baseline registry.

## Limitations before submission

- Repeat the remaining sweeps in independent processes and report confidence
  intervals; the primary hybrid MLP currently has three process repetitions.
- Randomize method order.
- Validate against an independent FP32 reference, not only HadaCore.
- Capture clocks, temperature, power, driver, compiler flags, and git status.
- Add latency/batch-size sweeps in addition to saturated throughput.
- Reproduce on A100, L40S, and H200.
- Benchmark pinned Tri Dao CUDA and FlagGems Triton baselines on identical inputs.
