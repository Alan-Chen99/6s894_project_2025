#!/usr/bin/env python3
"""Benchmark fused bidirectional RHT block-FP8 storage in TE training Linear."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

sys.path.insert(0, str(Path("paper/rht").resolve()))
from rht16_te_block import rht16_te_block_autograd, rht16_te_block_tensor  # noqa: E402
from rht16_triton import (  # noqa: E402
    reference_matrix,
    rht16,
    rht16_autograd,
    rht16_transpose,
)


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


def measure_paired(fn_a, fn_b, warmup: int, iterations: int):
    """Interleave two methods to reduce clock and order bias."""
    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    samples = [[], []]
    rng = random.Random(1234)
    for _ in range(iterations):
        order = (0, 1) if rng.random() < 0.5 else (1, 0)
        for index in order:
            fn = fn_a if index == 0 else fn_b
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))
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
        check = torch.randn(1024, 16, device="cuda", dtype=dtype)
        matrix = reference_matrix(device=check.device, dtype=dtype)
        transposed = rht16_transpose(check)
        roundtrip = rht16_transpose(rht16(check))
        correctness.append(
            {
                "dtype": str(dtype).removeprefix("torch."),
                "transpose_max_error": float((transposed - check @ matrix.T).abs().max()),
                "roundtrip_max_error": float((roundtrip - check).abs().max()),
            }
        )
    check_x = torch.randn(
        256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    check_layer = te.Linear(
        256, 256, bias=False, params_dtype=torch.bfloat16, device="cuda"
    )
    check_q = rht16_te_block_autograd(check_x)
    with te.autocast(enabled=True, recipe=block_recipe):
        check_y = check_layer(check_q)
    check_y.backward(torch.randn_like(check_y))
    torch.cuda.synchronize()
    autograd_correctness = {
        "output_finite": bool(torch.isfinite(check_y).all()),
        "input_grad_present": check_x.grad is not None,
        "input_grad_finite": bool(torch.isfinite(check_x.grad).all()),
        "weight_grad_present": check_layer.weight.grad is not None,
        "weight_grad_finite": bool(torch.isfinite(check_layer.weight.grad).all()),
    }
    del check_x, check_layer, check_q, check_y
    results = []
    for shape_name in args.shapes:
        m, k, n = SHAPES[shape_name]
        torch.manual_seed(1234)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        layer = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
        quantizer = te.Float8BlockQuantizer(
            te.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=1,
        )

        def separate_quantized_input():
            rotated = rht16(x.reshape(-1, 16)).reshape_as(x)
            q = quantizer(rotated)
            q.requires_grad_(True)
            return q

        def fused_quantized_input():
            q = rht16_te_block_tensor(x, quantizer, columnwise=True)
            q.requires_grad_(True)
            return q

        def separate_autograd_input():
            return rht16_autograd(x)

        def fused_autograd_input():
            return rht16_te_block_autograd(x)

        def train_step(make_input):
            layer.zero_grad(set_to_none=True)
            x.grad = None
            q = make_input()
            with te.autocast(enabled=True, recipe=block_recipe):
                y = layer(q)
            y.backward(dy)
            return y, x.grad, layer.weight.grad

        separate_pre, fused_pre = measure_paired(
            separate_quantized_input, fused_quantized_input, args.warmup, args.iterations
        )
        separate_train, fused_train = measure_paired(
            lambda: train_step(separate_autograd_input),
            lambda: train_step(fused_autograd_input),
            args.warmup,
            args.iterations,
        )
        samples_by_case = {
            "separate_preprocess": separate_pre,
            "fused_preprocess": fused_pre,
            "separate_train_pipeline": separate_train,
            "fused_train_pipeline": fused_train,
        }
        medians = {name: statistics.median(samples) for name, samples in samples_by_case.items()}
        row = {
            "shape": shape_name,
            "m": m,
            "k": k,
            "n": n,
            **{f"{name}_ms": value for name, value in medians.items()},
            "preprocess_fusion_speedup": medians["separate_preprocess"]
            / medians["fused_preprocess"],
            "train_pipeline_fusion_speedup": medians["separate_train_pipeline"]
            / medians["fused_train_pipeline"],
            "samples_ms": samples_by_case,
        }
        results.append(row)
        print(
            f"{shape_name:12s} preprocess {row['preprocess_fusion_speedup']:.3f}x "
            f"train-pipeline {row['train_pipeline_fusion_speedup']:.3f}x",
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
        "operation": "fused H16 RHT + TE rowwise/columnwise block FP8 + fprop/dgrad/wgrad + inverse RHT",
        "correctness": correctness,
        "autograd_correctness": autograd_correctness,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "method_order": "interleaved randomized pairs, seed 1234",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
