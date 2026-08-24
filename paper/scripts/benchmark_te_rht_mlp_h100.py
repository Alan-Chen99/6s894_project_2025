#!/usr/bin/env python3
"""Benchmark reusable fused RHT/FP8 activations in a Llama-style TE MLP."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.ops.basic import SwiGLU

sys.path.insert(0, str(Path("paper/rht").resolve()))
from rht16_te_block import rht16_te_block_autograd  # noqa: E402
from rht16_triton import rht16, rht16_autograd  # noqa: E402


def measure_paired(fn_a, fn_b, warmup: int, iterations: int):
    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    samples = [[], []]
    rng = random.Random(1234)
    for _ in range(iterations):
        for index in ((0, 1) if rng.random() < 0.5 else (1, 0)):
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
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--fc1-layout", choices=("combined", "split"), default="combined")
    parser.add_argument("--paired-weights", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile-method", choices=("plain", "separate", "fused"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m, k, n = args.tokens, args.hidden, args.intermediate
    if any(value % 128 for value in (m, k, n)):
        raise ValueError("tokens, hidden, and intermediate must be divisible by 128")

    torch.manual_seed(1234)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    if args.fc1_layout == "combined":
        fc1 = te.Linear(k, 2 * n, bias=False, params_dtype=torch.bfloat16, device="cuda")
        swiglu = SwiGLU()
        gate = up = None
        first_layers = (fc1,)
    else:
        fc1 = None
        swiglu = None
        gate = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
        up = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
        first_layers = (gate, up)
    down = te.Linear(n, k, bias=False, params_dtype=torch.bfloat16, device="cuda")
    layers = (*first_layers, down)
    if args.paired_weights:
        with torch.no_grad():
            for layer in layers:
                rotated = rht16(layer.weight.reshape(-1, 16)).reshape_as(layer.weight)
                layer.weight.copy_(rotated)
    block_recipe = recipe.Float8BlockScaling()

    def clear_grads() -> None:
        x.grad = None
        for layer in layers:
            layer.zero_grad(set_to_none=True)

    def mlp(make_input) -> torch.Tensor:
        qx = make_input(x)
        with te.autocast(enabled=True, recipe=block_recipe):
            if fc1 is not None:
                hidden = swiglu(fc1(qx))
            else:
                # The split layout intentionally reuses one quantized input.
                gate_out, up_out = gate(qx), up(qx)
                hidden = F.silu(gate_out) * up_out
        qh = make_input(hidden.contiguous())
        with te.autocast(enabled=True, recipe=block_recipe):
            return down(qh)

    def separate_input(value: torch.Tensor):
        return rht16_autograd(value)

    def fused_input(value: torch.Tensor):
        return rht16_te_block_autograd(value)

    def plain_input(value: torch.Tensor):
        return value

    def forward_step(make_input) -> torch.Tensor:
        return mlp(make_input)

    def train_step(make_input) -> torch.Tensor:
        clear_grads()
        y = mlp(make_input)
        y.backward(dy)
        return y

    if args.profile_method is not None:
        methods = {
            "plain": plain_input,
            "separate": separate_input,
            "fused": fused_input,
        }
        selected = methods[args.profile_method]
        for _ in range(args.warmup):
            train_step(selected)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(
            f"{args.profile_method}_mlp_{args.fc1_layout}_{m}_{k}_{n}"
        )
        train_step(selected)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    separate_forward, fused_forward = measure_paired(
        lambda: forward_step(separate_input),
        lambda: forward_step(fused_input),
        args.warmup,
        args.iterations,
    )
    separate_train, fused_train = measure_paired(
        lambda: train_step(separate_input),
        lambda: train_step(fused_input),
        args.warmup,
        args.iterations,
    )
    plain_forward, fused_control_forward = measure_paired(
        lambda: forward_step(plain_input),
        lambda: forward_step(fused_input),
        args.warmup,
        args.iterations,
    )
    plain_train, fused_control_train = measure_paired(
        lambda: train_step(plain_input),
        lambda: train_step(fused_input),
        args.warmup,
        args.iterations,
    )
    medians = {
        "separate_forward_ms": statistics.median(separate_forward),
        "fused_forward_ms": statistics.median(fused_forward),
        "separate_train_ms": statistics.median(separate_train),
        "fused_train_ms": statistics.median(fused_train),
        "plain_te_forward_ms": statistics.median(plain_forward),
        "fused_control_forward_ms": statistics.median(fused_control_forward),
        "plain_te_train_ms": statistics.median(plain_train),
        "fused_control_train_ms": statistics.median(fused_control_train),
    }
    check_y = train_step(fused_input)
    torch.cuda.synchronize()
    correctness = {
        "output_finite": bool(torch.isfinite(check_y).all()),
        "input_grad_present": x.grad is not None,
        "input_grad_finite": bool(torch.isfinite(x.grad).all()),
        "weight_grads_present": all(layer.weight.grad is not None for layer in layers),
        "weight_grads_finite": all(
            bool(torch.isfinite(layer.weight.grad).all()) for layer in layers
        ),
    }
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton_version": __import__("triton").__version__,
        "transformer_engine_version": transformer_engine.__version__,
        "operation": "Llama-style gate/up/down TE MLP with reusable RHT block-FP8 inputs",
        "shape": {"tokens": m, "hidden": k, "intermediate": n},
        "fc1_layout": args.fc1_layout,
        "paired_weights": args.paired_weights,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "method_order": "interleaved randomized pairs, seed 1234",
        "correctness": correctness,
        **medians,
        "forward_fusion_speedup": medians["separate_forward_ms"]
        / medians["fused_forward_ms"],
        "train_fusion_speedup": medians["separate_train_ms"]
        / medians["fused_train_ms"],
        "fused_vs_plain_forward": medians["plain_te_forward_ms"]
        / medians["fused_control_forward_ms"],
        "fused_vs_plain_train": medians["plain_te_train_ms"]
        / medians["fused_control_train_ms"],
        "samples_ms": {
            "separate_forward": separate_forward,
            "fused_forward": fused_forward,
            "separate_train": separate_train,
            "fused_train": fused_train,
            "plain_te_forward": plain_forward,
            "fused_control_forward": fused_control_forward,
            "plain_te_train": plain_train,
            "fused_control_train": fused_control_train,
        },
        "scope_note": (
            "The RHT methods use weights rotated on each linear's input axis, so their "
            "forward dataflow is paired with the activation rotation. Weights are rotated "
            "once outside timing; optimizer-state equivalence and convergence are not yet tested. "
            "The no-rotation TE control is throughput-only."
            if args.paired_weights
            else "Performance-only integration benchmark without paired weight rotations."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"forward {payload['forward_fusion_speedup']:.3f}x; "
        f"train {payload['train_fusion_speedup']:.3f}x; "
        f"vs-plain forward {payload['fused_vs_plain_forward']:.3f}x; "
        f"vs-plain train {payload['fused_vs_plain_train']:.3f}x"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
