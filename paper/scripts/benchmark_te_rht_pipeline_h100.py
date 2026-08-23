#!/usr/bin/env python3
"""Measure the incremental cost of H16 RHT before Transformer Engine Linear."""

from __future__ import annotations

import argparse
import contextlib
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
from rht16_triton import rht16  # noqa: E402


SHAPES = {
    "square_4096": (4096, 4096, 4096),
    "square_8192": (4096, 8192, 8192),
    "llama_up": (4096, 4096, 11008),
    "llama_down": (4096, 11008, 4096),
}


def timed(fn, warmup: int, iterations: int) -> list[float]:
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

    recipes = [
        ("bf16", None),
        ("fp8_delayed", recipe.DelayedScaling()),
        ("fp8_current", recipe.Float8CurrentScaling()),
        ("fp8_block", recipe.Float8BlockScaling()),
    ]
    results = []
    for shape_name in args.shapes:
        m, k, n = SHAPES[shape_name]
        for recipe_name, fp8_recipe in recipes:
            torch.manual_seed(1234)
            layer = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

            def linear(inp):
                ctx = contextlib.nullcontext() if fp8_recipe is None else te.autocast(
                    enabled=True, recipe=fp8_recipe
                )
                with ctx:
                    return layer(inp)

            baseline = timed(lambda: linear(x), args.warmup, args.iterations)
            rht_only = timed(lambda: rht16(x.reshape(-1, 16)), args.warmup, args.iterations)
            pipeline = timed(
                lambda: linear(rht16(x.reshape(-1, 16)).reshape_as(x)),
                args.warmup,
                args.iterations,
            )
            base_ms = statistics.median(baseline)
            rht_ms = statistics.median(rht_only)
            pipe_ms = statistics.median(pipeline)
            row = {
                "shape": shape_name,
                "recipe": recipe_name,
                "m": m,
                "k": k,
                "n": n,
                "linear_ms": base_ms,
                "rht_ms": rht_ms,
                "rht_linear_ms": pipe_ms,
                "incremental_ms": pipe_ms - base_ms,
                "pipeline_overhead_fraction": pipe_ms / base_ms - 1.0,
                "linear_samples_ms": baseline,
                "rht_samples_ms": rht_only,
                "pipeline_samples_ms": pipeline,
            }
            results.append(row)
            print(
                f"{shape_name:12s} {recipe_name:12s} linear={base_ms:.4f} ms "
                f"rht={rht_ms:.4f} ms pipeline={pipe_ms:.4f} ms "
                f"overhead={100 * row['pipeline_overhead_fraction']:.1f}%",
                flush=True,
            )

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformer_engine_version": transformer_engine.__version__,
        "operation": "H16 fixed-sign RHT followed by TE Linear fprop",
        "caveat": "Pipeline cost experiment; not a TE quantized-tensor integration or training result.",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
