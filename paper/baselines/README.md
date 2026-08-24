# External baseline registry

All external baselines must be pinned to an immutable commit, compiled for the
same GPU, and invoked with the same normalization, dtype, shape, and stream.
Compilation/JIT time is excluded from steady-state kernel timing but reported
separately.

## Tri Dao CUDA FWHT

- Repository: https://github.com/Dao-AILab/fast-hadamard-transform
- Version: `v1.1.0.post2`
- Commit: `e7706faf8d1c3b9f241e36860640ad1dac644ede`
- License: BSD-3-Clause
- Interface: `fast_hadamard_transform.hadamard_transform`
- Supported types: FP32, FP16, BF16
- Supported dimensions: through 32768, including implicit zero padding
- Upstream A100 claim: FP16/BF16 takes 1.0x memcpy through 512, at most 1.2x
  memcpy for 512--8192, 1.3x at 16384, and 1.8x at 32768.
- Status: pinned, compiled for H100 (`sm_90`), correctness-checked, and
  benchmarked. Raw standalone results are in
  `../results/raw/h100_sm90_dao_comparison_2026-08-23.json`. The H16 RHT--TE
  application comparison uses the same unmodified upstream kernel, rebuilt
  only for the PyTorch 2.8.0 ABI with SM90-only code generation. A separate
  stock PyTorch multiply supplies the random signs. Its summary is
  `../results/h100_nvl_dao_te_external_rht_summary.csv`.

This is the implementation referred to as "Tri Dao's" in the project
discussion. Our measured H100 speedup reaches 2.36x FP16 and 2.15x BF16 at
dimension 32768.

## FlagGems Triton FWHT

- Repository: https://github.com/flagos-ai/FlagGems
- Candidate stable version: `v5.0.2`
- Stable commit: `30bfcc0735b8ebd48bd7339ef752ec64df69663f`
- Current master observed: `ed2508bcb5a03000e9774734201d840ba362cd11`
- License: Apache-2.0
- Kernel: `src/flag_gems/ops/hadamard_transform.py`
- Implementation: Triton; source cites Dao-AILab's CUDA FWHT as its reference.
- Status: pinned and benchmarked directly from source with Triton 3.5.1;
  correctness passes at every size/dtype. Raw results are in
  `../../results/raw/h100_sm90_flaggems_comparison_2026-08-23.json`.

Before benchmarking, verify that the stable kernel supports the complete
256--32768 size range and both FP16/BF16. If not, report only its supported
domain rather than silently padding or changing the operation. The pinned
version supports the full tested 256--32768 power-of-two domain.

## arthurfeeney Triton FWHT

- Repository: https://github.com/arthurfeeney/fwht
- Commit: `9fcb9fe8dfac000bbebed001809b02b49a55dcdc`
- Upstream status: work in progress; no license file at the pinned commit
- Interface: `fwht.fast_hadamard_transform`
- Implementation: Triton `tl.dot` tensor-core decomposition with optional
  normalization and in-place execution
- Maximum supported dimension: 32768
- Status: pinned and benchmarked on H100 with a documented local BF16
  compatibility patch. Unmodified upstream requests an unsupported BF16
  `tl.dot` output under Triton 3.5.1; our patch uses FP32 dot output and casts
  back at the original BF16 stage boundaries. Correctness passes for both
  FP16 and BF16 at every tested size. Raw results are in
  `../../results/raw/h100_sm90_arthur_comparison_2026-08-23.json`.

This is a useful secondary research baseline, but the WIP status and absence of
an explicit license make it less suitable than Tri Dao and FlagGems as the sole
primary comparison.

## Excluded from the primary standalone-FWHT table

Fixed-H16 random-Hadamard kernels fused with transpose, amax, or quantization
(for example TransformerEngine/PyTorch AO RHT paths) compute a different
operation and should appear only in a separate application/fusion experiment.
