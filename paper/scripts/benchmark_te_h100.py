#!/usr/bin/env python3
"""Benchmark Transformer Engine BF16 and FP8 Linear paths on one GPU."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe


SHAPES = {
    "square_4096": (4096, 4096, 4096),
    "square_8192": (4096, 8192, 8192),
    "llama_up": (4096, 4096, 11008),
    "llama_down": (4096, 11008, 4096),
}


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def quant_context(fp8_recipe):
    if fp8_recipe is None:
        return contextlib.nullcontext()
    return te.autocast(enabled=True, recipe=fp8_recipe)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def measure(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def run_case(name: str, fp8_recipe, m: int, k: int, n: int, mode: str, warmup: int, iterations: int):
    torch.manual_seed(1234)
    layer = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=mode == "train")
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) if mode == "train" else None

    def step():
        if mode == "train":
            layer.zero_grad(set_to_none=True)
            x.grad = None
        with quant_context(fp8_recipe):
            y = layer(x)
        if mode == "train":
            y.backward(dy)
        return y

    try:
        samples = measure(step, warmup, iterations)
        median = statistics.median(samples)
        # One GEMM in fprop; fprop+dgrad+wgrad in train mode.
        flop_multiplier = 3 if mode == "train" else 1
        tflops = flop_multiplier * 2.0 * m * k * n / (median * 1e9)
        return {
            "recipe": name,
            "mode": mode,
            "m": m,
            "k": k,
            "n": n,
            "status": "ok",
            "median_ms": median,
            "mean_ms": statistics.mean(samples),
            "p10_ms": percentile(samples, 0.10),
            "p90_ms": percentile(samples, 0.90),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "effective_tflops": tflops,
            "samples_ms": samples,
        }
    except Exception as exc:
        torch.cuda.synchronize()
        return {
            "recipe": name,
            "mode": mode,
            "m": m,
            "k": k,
            "n": n,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=["square_4096"])
    parser.add_argument("--modes", nargs="+", choices=["fprop", "train"], default=["fprop", "train"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    recipes = [
        ("bf16", None),
        ("fp8_delayed", recipe.DelayedScaling()),
        ("fp8_current", recipe.Float8CurrentScaling()),
        ("fp8_block", recipe.Float8BlockScaling()),
    ]
    availability = {
        "fp8": te.is_fp8_available(),
        "fp8_block": te.is_fp8_block_scaling_available(),
        "mxfp8": te.is_mxfp8_available(),
        "nvfp4": te.is_nvfp4_available(),
    }
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformer_engine_version": transformer_engine.__version__,
        "availability": availability,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": [],
    }
    print(json.dumps(availability, default=str))
    for shape_name in args.shapes:
        m, k, n = SHAPES[shape_name]
        for mode in args.modes:
            for recipe_name, fp8_recipe in recipes:
                result = run_case(recipe_name, fp8_recipe, m, k, n, mode, args.warmup, args.iterations)
                result["shape"] = shape_name
                payload["results"].append(result)
                if result["status"] == "ok":
                    print(f"{shape_name:12s} {mode:5s} {recipe_name:12s} "
                          f"{result['median_ms']:9.4f} ms {result['effective_tflops']:8.1f} TFLOP/s")
                else:
                    print(f"{shape_name:12s} {mode:5s} {recipe_name:12s} ERROR {result['error']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
