#!/usr/bin/env python3
"""Compare the final dynamic RHT path directly with ordinary TE block FP8."""

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
from dynamic_weight import (  # noqa: E402
    DynamicQuantizedWeightBridge,
    DynamicWeightBridge,
    FusedRHTAdamW,
    clear_all_working_grads,
    map_all_grads,
    materialize_all,
)
from rht16_te_block import (  # noqa: E402
    rht16_te_block_autograd,
    swiglu_rht16_te_block_autograd,
)


def measure_paired(fn_a, fn_b, warmup: int, iterations: int):
    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    samples = [[], []]
    rng = random.Random(894)
    for _ in range(iterations):
        order = (0, 1) if rng.random() < 0.5 else (1, 0)
        for index in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            (fn_a if index == 0 else fn_b)()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))
    return samples


def measure_backward_paired(prepare_a, prepare_b, dy, warmup: int, iterations: int):
    for _ in range(warmup):
        prepare_a().backward(dy)
        prepare_b().backward(dy)
    torch.cuda.synchronize()
    samples = [[], []]
    rng = random.Random(895)
    for _ in range(iterations):
        outputs = [prepare_a(), prepare_b()]
        order = (0, 1) if rng.random() < 0.5 else (1, 0)
        for index in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs[index].backward(dy)
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))
    return samples


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().float()
    expected = expected.detach().float()
    return float((actual - expected).norm() / expected.norm())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m, k, n = args.tokens, args.hidden, args.intermediate
    if any(value % 128 for value in (m, k, n)):
        raise ValueError("all dimensions must be divisible by 128")

    torch.manual_seed(1234)
    initial_w1 = torch.randn(2 * n, k, device="cuda") / (k**0.5)
    initial_w2 = torch.randn(k, n, device="cuda") / (n**0.5)
    block_recipe = recipe.Float8BlockScaling()

    plain_fc1 = te.Linear(
        k, 2 * n, bias=False, params_dtype=torch.bfloat16, device="cuda"
    )
    plain_down = te.Linear(
        n, k, bias=False, params_dtype=torch.bfloat16, device="cuda"
    )
    plain_bridges = [
        DynamicWeightBridge.from_working_weight(
            plain_fc1.weight, rotated=False, initial=initial_w1
        ),
        DynamicWeightBridge.from_working_weight(
            plain_down.weight, rotated=False, initial=initial_w2
        ),
    ]
    with te.quantized_model_init(enabled=True, recipe=block_recipe):
        fused_fc1 = te.Linear(
            k, 2 * n, bias=False, params_dtype=torch.bfloat16, device="cuda"
        )
        fused_down = te.Linear(
            n, k, bias=False, params_dtype=torch.bfloat16, device="cuda"
        )
    fused_bridges = [
        DynamicQuantizedWeightBridge.attach(fused_fc1, initial_w1),
        DynamicQuantizedWeightBridge.attach(fused_down, initial_w2),
    ]
    materialize_all(plain_bridges)
    materialize_all(fused_bridges)

    plain_optimizer = torch.optim.AdamW(
        [bridge.master for bridge in plain_bridges],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=True,
    )
    fused_optimizer = FusedRHTAdamW(
        fused_bridges,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    plain_swiglu = SwiGLU()
    x_plain = torch.randn(
        m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_fused = x_plain.detach().clone().requires_grad_(True)
    dy = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    def plain_mlp():
        with te.autocast(enabled=True, recipe=block_recipe):
            return plain_down(plain_swiglu(plain_fc1(x_plain)))

    def fused_mlp():
        with te.autocast(enabled=True, recipe=block_recipe):
            z = fused_fc1(rht16_te_block_autograd(x_fused))
            hidden = swiglu_rht16_te_block_autograd(z)
            return fused_down(hidden)

    def plain_forward():
        materialize_all(plain_bridges)
        return plain_mlp()

    def fused_forward():
        materialize_all(fused_bridges)
        return fused_mlp()

    def plain_forward_backward():
        x_plain.grad = None
        clear_all_working_grads(plain_bridges)
        plain_forward().backward(dy)

    def fused_forward_backward():
        x_fused.grad = None
        clear_all_working_grads(fused_bridges)
        fused_forward().backward(dy)

    def prepare_plain_backward():
        x_plain.grad = None
        clear_all_working_grads(plain_bridges)
        materialize_all(plain_bridges)
        return plain_mlp()

    def prepare_fused_backward():
        x_fused.grad = None
        clear_all_working_grads(fused_bridges)
        materialize_all(fused_bridges)
        return fused_mlp()

    def plain_full_step():
        plain_optimizer.zero_grad(set_to_none=True)
        plain_forward_backward()
        map_all_grads(plain_bridges)
        plain_optimizer.step()

    def fused_full_step():
        fused_optimizer.zero_grad(set_to_none=True)
        fused_forward_backward()
        fused_optimizer.step()

    # Numerical comparison before either optimizer changes the common source.
    plain_y = plain_forward()
    fused_y = fused_forward()
    with torch.no_grad():
        reference_z = x_plain.detach() @ initial_w1.to(torch.bfloat16).T
        reference_gate, reference_up = reference_z.chunk(2, dim=-1)
        bf16_reference_y = (
            (F.silu(reference_gate) * reference_up)
            @ initial_w2.to(torch.bfloat16).T
        )
    numerical = {
        "fused_vs_plain_relative_l2": relative_l2(fused_y, plain_y),
        "plain_vs_bf16_relative_l2": relative_l2(plain_y, bf16_reference_y),
        "fused_vs_bf16_relative_l2": relative_l2(fused_y, bf16_reference_y),
    }

    forward_samples = measure_paired(
        plain_forward, fused_forward, args.warmup, args.iterations
    )
    backward_samples = measure_backward_paired(
        prepare_plain_backward,
        prepare_fused_backward,
        dy,
        args.warmup,
        args.iterations,
    )
    forward_backward_samples = measure_paired(
        plain_forward_backward,
        fused_forward_backward,
        args.warmup,
        args.iterations,
    )

    # Initialize optimizer states before measuring complete steps.
    plain_full_step()
    fused_full_step()
    full_step_samples = measure_paired(
        plain_full_step, fused_full_step, args.warmup, args.iterations
    )
    torch.cuda.synchronize()

    plain_forward_ms = statistics.median(forward_samples[0])
    fused_forward_ms = statistics.median(forward_samples[1])
    plain_backward_ms = statistics.median(backward_samples[0])
    fused_backward_ms = statistics.median(backward_samples[1])
    plain_forward_backward_ms = statistics.median(forward_backward_samples[0])
    fused_forward_backward_ms = statistics.median(forward_backward_samples[1])
    plain_full_step_ms = statistics.median(full_step_samples[0])
    fused_full_step_ms = statistics.median(full_step_samples[1])
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformer_engine_version": transformer_engine.__version__,
        "shape": {"tokens": m, "hidden": k, "intermediate": n},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "numerical": numerical,
        "forward": {
            "plain_te_ms": plain_forward_ms,
            "fused_rht_ms": fused_forward_ms,
            "speedup": plain_forward_ms / fused_forward_ms,
        },
        "forward_backward": {
            "plain_te_ms": plain_forward_backward_ms,
            "fused_rht_ms": fused_forward_backward_ms,
            "speedup": plain_forward_backward_ms / fused_forward_backward_ms,
        },
        "backward": {
            "plain_te_ms": plain_backward_ms,
            "fused_rht_ms": fused_backward_ms,
            "speedup": plain_backward_ms / fused_backward_ms,
        },
        "full_step": {
            "plain_te_ms": plain_full_step_ms,
            "fused_rht_ms": fused_full_step_ms,
            "speedup": plain_full_step_ms / fused_full_step_ms,
        },
        "samples_ms": {
            "plain_te_forward": forward_samples[0],
            "fused_rht_forward": forward_samples[1],
            "plain_te_backward": backward_samples[0],
            "fused_rht_backward": backward_samples[1],
            "plain_te_forward_backward": forward_backward_samples[0],
            "fused_rht_forward_backward": forward_backward_samples[1],
            "plain_te_full_step": full_step_samples[0],
            "fused_rht_full_step": full_step_samples[1],
        },
        "scope_note": (
            "Both paths keep independently owned original-basis FP32 AdamW "
            "masters. Plain TE materializes BF16 weights and lets TE perform "
            "ordinary block-FP8 quantization. The RHT path writes paired "
            "transformed weights directly into TE 2D block-FP8 storage and "
            "fuses inverse RHT with AdamW. Forward and forward+backward include "
            "weight materialization; only full_step includes AdamW."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "numerical": numerical,
        "forward": payload["forward"],
        "forward_backward": payload["forward_backward"],
        "backward": payload["backward"],
        "full_step": payload["full_step"],
    }, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
