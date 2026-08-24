#!/usr/bin/env python3
"""Benchmark fused RHT+TE block-128 quantization against separate kernels."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

sys.path.insert(0, str(Path("paper/rht").resolve()))
from rht16_te_block import rht16_te_block_tensor  # noqa: E402
from rht16_triton import rht16  # noqa: E402


SHAPES = {
    "square_4096": (4096, 4096, 4096),
    "square_8192": (4096, 8192, 8192),
    "llama_up": (4096, 4096, 11008),
    "llama_down": (4096, 11008, 4096),
}


def measure(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    block_recipe = recipe.Float8BlockScaling()
    correctness = []
    for dtype in (torch.float16, torch.bfloat16):
        check = torch.randn(256, 256, device="cuda", dtype=dtype)
        truth = rht16(check.reshape(-1, 16)).reshape_as(check)
        quantizer = te.Float8BlockQuantizer(
            te.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            block_scaling_dim=1,
        )
        fused = rht16_te_block_tensor(check, quantizer)
        reference = quantizer(truth)
        fused_dequant = fused.dequantize().float()
        reference_dequant = reference.dequantize().float()
        truth_float = truth.float()
        correctness.append(
            {
                "dtype": str(dtype).removeprefix("torch."),
                "scale_max_error": float(
                    (fused._rowwise_scale_inv - reference._rowwise_scale_inv).abs().max()
                ),
                "fp8_byte_agreement": float(
                    (fused._rowwise_data == reference._rowwise_data).float().mean()
                ),
                "fused_reconstruction_mse": float(
                    ((fused_dequant - truth_float) ** 2).mean()
                ),
                "reference_reconstruction_mse": float(
                    ((reference_dequant - truth_float) ** 2).mean()
                ),
            }
        )
    results = []
    for shape_name in args.shapes:
        m, k, n = SHAPES[shape_name]
        torch.manual_seed(1234)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        layer = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
        input_quantizer = te.Float8BlockQuantizer(
            te.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            block_scaling_dim=1,
        )

        def linear(inp):
            with te.autocast(enabled=True, recipe=block_recipe):
                return layer(inp)

        def separate_quantized():
            rotated = rht16(x.reshape(-1, 16)).reshape_as(x)
            return input_quantizer(rotated)

        def fused_quantized():
            return rht16_te_block_tensor(x, input_quantizer)

        cases = {
            "te_block_plain": lambda: linear(x),
            "separate_rht_quantize": separate_quantized,
            "fused_rht_quantize": fused_quantized,
            "separate_pipeline": lambda: linear(separate_quantized()),
            "fused_pipeline": lambda: linear(fused_quantized()),
        }
        medians = {}
        case_samples = {}
        for name, fn in cases.items():
            samples = measure(fn, args.warmup, args.iterations)
            case_samples[name] = samples
            medians[name] = statistics.median(samples)
        row = {
            "shape": shape_name,
            "m": m,
            "k": k,
            "n": n,
            **{f"{name}_ms": value for name, value in medians.items()},
            "quantizer_fusion_speedup": medians["separate_rht_quantize"]
            / medians["fused_rht_quantize"],
            "pipeline_fusion_speedup": medians["separate_pipeline"]
            / medians["fused_pipeline"],
            "fused_overhead_vs_te_plain": medians["fused_pipeline"]
            / medians["te_block_plain"]
            - 1.0,
            "samples_ms": case_samples,
        }
        results.append(row)
        print(
            f"{shape_name:12s} quantize {row['quantizer_fusion_speedup']:.3f}x "
            f"pipeline {row['pipeline_fusion_speedup']:.3f}x "
            f"fused-overhead {100 * row['fused_overhead_vs_te_plain']:.1f}%",
            flush=True,
        )

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton_version": __import__("triton").__version__,
        "transformer_engine_version": transformer_engine.__version__,
        "operation": "fused H16 RHT + TE-compatible block-128 E4M3 quantization",
        "correctness": correctness,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
