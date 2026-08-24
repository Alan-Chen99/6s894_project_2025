# Tensor-core FWHT research board

Updated: 2026-08-23

This is the repo-native ICLR 2027 project canvas. Keep raw measurements under
`paper/results/raw/`, derived tables under `paper/results/`, and mark claims as
paper-ready only after independent repetitions and provenance capture.

## Current headline

On H100, the kernel reaches roughly 3 TB/s effective bandwidth and is
competitive with every tested standalone FWHT baseline. The strongest wins are
at large transform sizes: up to 1.59x over HadaCore, 2.36x over Tri Dao, 2.25x
over FlagGems, and 1.28x/1.36x over patched arthurfeeney/fwht for FP16/BF16.

## Workstreams

| Workstream | Status | Next executable milestone | Required GPU |
|---|---|---|---|
| H100 standalone baselines | Results captured | Repeat randomized-order runs and confidence intervals | H100 |
| Instruction profiling | Results captured | Add opcode/SASS categorization and independent profile repetition | H100 |
| Random Hadamard transform | Prototype measured | Replace Triton prototype with/integrate into the CUDA path | H100 |
| H16 RHT + quantization | Single-pass forward/backward adapter profiled | Test model-derived tensors | H100 initially |
| Native FP8 | Linear baseline measured | Test model-derived tensors and quantify RHT impact | H100 |
| MXFP8 | Hardware blocked | Benchmark block-32 E8M0-scaled path | SM100+ Blackwell |
| NVFP4 | Hardware blocked | Benchmark H16 RHT + block-16 E4M3 scaling + E2M1 output | SM100+ Blackwell |
| End-to-end training | Dynamic Adam path + controlled convergence captured | Optimize TE primary-weight runtime; scale to language modeling | H100/B200 |

Transformer Engine 2.18.0 is staged under `paper/baselines/te_runtime/` with a
locally compiled PyTorch CUDA binding. It has passed a BF16 `te.Linear` smoke
test with PyTorch 2.9.1+cu128 on H100 allocation `21044277` (`node2900`). A
separate `.venv-cu129` with PyTorch 2.8.0+cu129 enables delayed, current, and
block FP8 scaling. Allocation `21044277` expired before the subsequent RHT-to-
TE pipeline-cost script could run; it was completed on allocation `21053440`.
The first timed TE linear comparison is saved under `paper/results/raw/` and
summarized in `paper/results/h100_sm90_transformer_engine_summary.csv`.
In a matched PyTorch 2.8.0+cu129 sweep, delayed-scaling FP8 reaches 1.32x fprop
and 1.66x fprop+dgrad+wgrad speedup over BF16 for the 4096x8192x8192 case. It
is slower for the smaller 4096-cubed fprop, demonstrating the quantization-
overhead crossover. The isolated CUDA 12.9 environment enables FP8 block
scaling: it reaches 1.17x and 1.34x on those two measurements, but trails
delayed scaling. The local FP8 fusion ablation remains a separate result.

The first RHT-to-TE forward pipeline measurement applies the fixed-sign H16
transform to BF16 activations before the same TE Linear. For square-8192,
delayed FP8 plus RHT is 1.19x faster than plain BF16 Linear and 1.30x faster
than BF16 plus RHT. For the Llama up projection those values are 1.13x and
1.27x. The down projection is approximately tied with plain BF16 (0.98x) but
1.13x faster than BF16 plus RHT. Square-4096 does not amortize the transform.
This is a forward pipeline-cost experiment, not yet a direct TE quantized-
tensor integration or an end-to-end training result.

The follow-up implements that direct Hopper adapter: one Triton kernel applies
eight H16 transforms per TE block, computes TE-compatible power-of-two block-
128 scales, writes E4M3 bytes in TE's rowwise layout, and wraps the buffers as
a `Float8BlockwiseQTensor` with no requantization kernel. Scales match TE's
reference exactly and reconstruction MSE is identical for FP16 and BF16. The
fused quantizer is 1.27--1.34x faster than separate RHT then TE quantization;
the complete TE Linear pipeline is 1.02--1.10x faster than the separate
pipeline. At the three larger shapes, fused RHT adds only 2.4--6.5% over a
plain block-FP8 TE Linear in the forward-only experiment.

The backward adapter writes TE's independently scaled columnwise FP8 view for
Wgrad and applies the exact transpose transform, $R^T=HS/4$, to Dgrad.
Dense-reference transpose error is zero for FP16/BF16, and TE produces finite
input and weight gradients. The final single-pass 128x128 writer computes H16
once, reduces along both axes, and emits both TE layouts. It matches the two-
writer implementation byte-for-byte and scale-for-scale. Two-view preprocessing
is 1.20--1.37x faster than separate RHT plus TE quantization. Forward ranges
from 1.01x at square-4096 to 1.12x for Llama down; the complete training step
ranges from 0.99x to 1.04x. Nsight shows a 9.6% reduction in the square-4096
GPU interval, while GEMM and inverse-RHT times are unchanged. These H100 NVL
timings must not be merged with the H100 80GB HBM3 tables.

A Llama-shaped MLP benchmark uses a combined 4096-to-22016 FC1, Transformer
Engine's fused SwiGLU, and a 11008-to-4096 down projection. Each weight is
paired with the activation RHT on its input axis. In an independent numerical
check, FP32 paired outputs and gradients agree within 4.5e-7--5.7e-7 relative
L2; BF16 differences are about 0.60--0.61%. TE block-FP8 paired RHT has
6.56--6.82% relative error against BF16, marginally below the corresponding
plain block-FP8 errors in all four measured quantities. Those timings rotate
weights once outside the measured region; the optimizer-correct dynamic
extension below is reported separately.

Folding SwiGLU directly into the 128x128 forward writer was slower on SM90
(268 us) because of register pressure. The retained hybrid uses TE's dedicated
SwiGLU forward kernel and fuses inverse H16 with dSwiGLU in backward. Nsight
shows that the fused backward kernel replaces 275.5 us of separate work with
220.9 us (-19.8%) and removes one launch. Across three independent processes,
median speedups over separate RHT plus TE are 1.054x forward, 1.020x backward,
and 1.027x for the complete step; the observed ranges are 1.053--1.059x,
1.009--1.021x, and 1.015--1.034x. At the high-expansion 3584-to-18944 shape
with 8192 tokens, forward/training speedups are 1.073x/1.032x.

The first model-shape sweep shows that gains depend more on FFN expansion and
token saturation than parameter width alone. A 3584-to-18944 high-expansion
shape improves from 1.020x/1.008x forward/training at 1024 tokens to
1.073x/1.032x at 8192 tokens. A 4096-to-14336 shape reaches 1.059x/1.021x,
while the larger 8192-to-28672 shape at 2048 tokens reaches only
1.025x/1.012x because GEMMs dominate. The current Hopper sweet spot is thus a
high expansion ratio with a sufficiently large token batch.

The dynamic-weight extension keeps FP32 AdamW masters and optimizer moments in
the original basis. Each step writes $WR$ directly into TE's native 128x128
2D block-FP8 weight storage, then maps BF16 Wgrad through $R^T$ into the FP32
master-gradient buffer. The fused writer matches TE's reference row/column
bytes and valid scales exactly; the optimizer check has 1.85e-9 gradient
relative L2 and an exactly matching AdamW update. On a controlled 120-step
teacher--student SwiGLU task, RHT block-FP8 finishes at 0.001562 MSE versus
0.001609 for ordinary block-FP8, so RHT adds no observed convergence regression
relative to the FP8 control. BF16 reaches 0.000268, exposing the FP8 quality
floor. At the 4096-to-11008 shape, directly quantized dynamic weights take
7.732 ms versus 7.814 ms for the interleaved BF16-working dynamic control
(1.011x faster). Nsight projects 6.106 versus 6.375 ms and 11 versus 13 GPU
operations because the fused path removes two TE weight-quantization kernels.
Both dynamic routes remain about 1.18--1.19x slower than their paired static
controls; these numbers exclude AdamW kernels and identify dynamic
materialization/gradient mapping as the remaining end-to-end cost.

## Low-precision decision

H100 natively supports ordinary FP8, so the practical Hopper experiment is
RHT + FP8 quantization, ideally fused with amax/scaling. MXFP8 and NVFP4 native
Tensor Core execution require SM100 or later. We can implement their dataflow
and numerical emulation on H100, but those runs must not be presented as native
low-precision throughput.

Suggested order:

1. Add a fixed random sign vector to the FWHT load path. A sign flip is exact
   and can be fused before the first tensor-core stage with no extra memory pass.
2. Implement the tiled H16 RHT used by Transformer Engine for FP4 Wgrad inputs.
   This is a different regime from our 256--32768 standalone sweep and needs a
   separate latency/fusion table.
3. On H100, fuse RHT with FP8 amax, scale, and cast. Compare separate versus
   fused pipelines and report quantization error/outlier reduction.
4. On Blackwell, replace the output stage with native MXFP8 and NVFP4 packing,
   then compare full linear forward/backward throughput in Transformer Engine.

Current H16 prototype result: plain RHT is within 2.3% of a dense H16 matmul
reference. Fusing RHT, per-16 amax/scale, and E4M3 output takes 0.262/0.264 ms
for 67.1M FP16/BF16 input elements, versus 1.578/1.580 ms for the eager
separate pipeline (6.01x/5.98x). This is a preliminary fusion ablation, not yet
a comparison against Transformer Engine or a native NVFP4 result.

## End-to-end experiment ladder

| Level | Measurement | Success criterion |
|---|---|---|
| Kernel | RHT alone; RHT+amax; RHT+quantize | Correctness and lower total pipeline time |
| Linear layer | Fprop, Dgrad, Wgrad separately | Net speedup after quantization overhead |
| Transformer block | Forward/backward step time and memory | Speedup survives non-GEMM operations |
| Small training run | Loss curve, gradient statistics, tokens/s | Captured: no regression versus block-FP8 control |
| Model scale | Time-to-quality and distributed communication | Reproducible end-to-end gain |

## Profiling protocol

Use the same shape, dtype, in-place semantics, clock state, and isolated GPU for
both implementations. Profile one launch after warmup; profiling hundreds of
launches distorts execution and produces redundant data.

```bash
export CUDA_ROOT=/orcd/software/core/001/pkg/cuda/12.9.1
export PYTHONPATH="$PWD/csrc:$PWD/HadaCore"

$CUDA_ROOT/bin/ncu \
  --nvtx --nvtx-include 'profile_ours_float16_n32768/' \
  --section SpeedOfLight --section InstructionStats \
  --section MemoryWorkloadAnalysis --section SourceCounters \
  --export paper/profiles/h100_ours_fp16_n32768 \
  .venv/bin/python paper/scripts/profile_fwht.py \
  --impl ours --dtype float16 --size 32768

$CUDA_ROOT/bin/ncu \
  --nvtx --nvtx-include 'profile_hadacore_float16_n32768/' \
  --section SpeedOfLight --section InstructionStats \
  --section MemoryWorkloadAnalysis --section SourceCounters \
  --export paper/profiles/h100_hadacore_fp16_n32768 \
  .venv/bin/python paper/scripts/profile_fwht.py \
  --impl hadacore --dtype float16 --size 32768
```

The instruction claim is based on comparable dynamic metrics, not SASS file
length. Across sizes 256, 4096, and 32768, ours executes 2.92x, 2.25x, and
3.47x fewer FP16 warp instructions; BF16 reductions are 2.79x, 2.02x, and
3.60x. Small transforms remain bandwidth/launch limited despite the reduction.
The next pass should categorize opcodes, spills, and major warp-stall reasons.

## Paper-claim checklist

- [x] Correctness against external implementations on H100
- [x] HadaCore, Tri Dao, FlagGems, and arthurfeeney comparisons
- [x] Raw JSON, pinned revisions, and environment provenance
- [ ] Independent repetitions with confidence intervals
- [ ] Randomized benchmark order and locked-clock metadata
- [x] Nsight instruction-count evidence at three sizes and both dtypes
- [x] RHT correctness and fused-pipeline benchmark
- [x] Native FP8 H100 kernel prototype
- [x] Transformer Engine FP8 linear-layer comparison
- [x] Original-basis AdamW gradient/state handling
- [x] Controlled 120-step convergence and throughput plots
- [ ] Native MXFP8/NVFP4 Blackwell experiment
- [ ] Language-model end-to-end training throughput and quality
- [ ] A100/H200/Blackwell architecture comparison

## Open decisions

- Primary application target: Transformer Engine NVFP4 Wgrad, generic FP8
  training, or inference quantization.
- Whether the paper centers on standalone FWHT architecture or the fused
  quantization/training pipeline.
- Blackwell allocation and software stack for native MXFP8/NVFP4 results.
