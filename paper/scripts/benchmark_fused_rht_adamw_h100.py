#!/usr/bin/env python3
"""Benchmark inverse-RHT fused directly into original-basis AdamW."""

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
from dynamic_weight import (  # noqa: E402
    DynamicQuantizedWeightBridge,
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
    rng = random.Random(2027)
    for _ in range(iterations):
        for index in ((0, 1) if rng.random() < 0.5 else (1, 0)):
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
    rng = random.Random(2028)
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

    def make_path():
        with te.quantized_model_init(enabled=True, recipe=block_recipe):
            fc1 = te.Linear(
                k, 2 * n, bias=False, params_dtype=torch.bfloat16, device="cuda"
            )
            down = te.Linear(
                n, k, bias=False, params_dtype=torch.bfloat16, device="cuda"
            )
        bridges = [
            DynamicQuantizedWeightBridge.attach(fc1, initial_w1),
            DynamicQuantizedWeightBridge.attach(down, initial_w2),
        ]
        materialize_all(bridges)
        return fc1, down, bridges

    reference_fc1, reference_down, reference_bridges = make_path()
    fused_fc1, fused_down, fused_bridges = make_path()
    reference_optimizer = torch.optim.AdamW(
        [bridge.master for bridge in reference_bridges],
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
    x_reference = torch.randn(
        m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_fused = x_reference.detach().clone().requires_grad_(True)
    dy = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    def mlp(x, fc1, down):
        with te.autocast(enabled=True, recipe=block_recipe):
            z = fc1(rht16_te_block_autograd(x))
            hidden = swiglu_rht16_te_block_autograd(z)
            return down(hidden)

    def reference_forward():
        materialize_all(reference_bridges)
        return mlp(x_reference, reference_fc1, reference_down)

    def fused_forward():
        materialize_all(fused_bridges)
        return mlp(x_fused, fused_fc1, fused_down)

    def prepare_reference_backward():
        x_reference.grad = None
        clear_all_working_grads(reference_bridges)
        return reference_forward()

    def prepare_fused_backward():
        x_fused.grad = None
        clear_all_working_grads(fused_bridges)
        return fused_forward()

    # The two phase paths are byte-for-byte identical before optimization.
    phase_output_relative_l2 = relative_l2(fused_forward(), reference_forward())
    forward_samples = measure_paired(
        reference_forward, fused_forward, args.warmup, args.iterations
    )
    backward_samples = measure_backward_paired(
        prepare_reference_backward,
        prepare_fused_backward,
        dy,
        args.warmup,
        args.iterations,
    )

    def reference_step():
        x_reference.grad = None
        reference_optimizer.zero_grad(set_to_none=True)
        clear_all_working_grads(reference_bridges)
        y = reference_forward()
        y.backward(dy)
        map_all_grads(reference_bridges)
        reference_optimizer.step()
        return y

    def fused_step():
        x_fused.grad = None
        fused_optimizer.zero_grad(set_to_none=True)
        y = fused_forward()
        y.backward(dy)
        fused_optimizer.step()
        return y

    # Establish update equivalence before repeated timing changes the two
    # independently owned states through slightly different FP32 reductions.
    reference_step()
    fused_step()
    reference_states = [
        reference_optimizer.state[bridge.master] for bridge in reference_bridges
    ]
    one_step_correctness = {
        "fc1_master_relative_l2": relative_l2(
            fused_bridges[0].master, reference_bridges[0].master
        ),
        "down_master_relative_l2": relative_l2(
            fused_bridges[1].master, reference_bridges[1].master
        ),
        "fc1_exp_avg_relative_l2": relative_l2(
            fused_optimizer.state[0]["exp_avg"], reference_states[0]["exp_avg"]
        ),
        "fc1_exp_avg_sq_relative_l2": relative_l2(
            fused_optimizer.state[0]["exp_avg_sq"], reference_states[0]["exp_avg_sq"]
        ),
    }

    reference_samples, fused_samples = measure_paired(
        reference_step, fused_step, args.warmup, args.iterations
    )

    post_full_timing_drift = {
        "fc1_master_relative_l2": relative_l2(
            fused_bridges[0].master, reference_bridges[0].master
        ),
        "down_master_relative_l2": relative_l2(
            fused_bridges[1].master, reference_bridges[1].master
        ),
        "fc1_exp_avg_relative_l2": relative_l2(
            fused_optimizer.state[0]["exp_avg"], reference_states[0]["exp_avg"]
        ),
        "fc1_exp_avg_sq_relative_l2": relative_l2(
            fused_optimizer.state[0]["exp_avg_sq"],
            reference_states[0]["exp_avg_sq"],
        ),
    }

    # Isolate the optimizer tail with fixed representative working gradients.
    reference_step()
    fused_step()

    def reference_optimizer_tail():
        reference_optimizer.zero_grad(set_to_none=True)
        map_all_grads(reference_bridges)
        reference_optimizer.step()

    def fused_optimizer_tail():
        fused_optimizer.step()

    reference_tail, fused_tail = measure_paired(
        reference_optimizer_tail,
        fused_optimizer_tail,
        args.warmup,
        args.iterations,
    )
    torch.cuda.synchronize()

    reference_ms = statistics.median(reference_samples)
    fused_ms = statistics.median(fused_samples)
    reference_forward_ms = statistics.median(forward_samples[0])
    fused_forward_ms = statistics.median(forward_samples[1])
    reference_backward_ms = statistics.median(backward_samples[0])
    fused_backward_ms = statistics.median(backward_samples[1])
    reference_tail_ms = statistics.median(reference_tail)
    fused_tail_ms = statistics.median(fused_tail)
    post_tail_timing_drift = {
        "fc1_master_relative_l2": relative_l2(
            fused_bridges[0].master, reference_bridges[0].master
        ),
        "down_master_relative_l2": relative_l2(
            fused_bridges[1].master, reference_bridges[1].master
        ),
        "fc1_exp_avg_relative_l2": relative_l2(
            fused_optimizer.state[0]["exp_avg"], reference_states[0]["exp_avg"]
        ),
        "fc1_exp_avg_sq_relative_l2": relative_l2(
            fused_optimizer.state[0]["exp_avg_sq"],
            reference_states[0]["exp_avg_sq"],
        ),
        "all_finite": bool(
            all(
                torch.isfinite(bridge.master).all()
                for bridge in (*reference_bridges, *fused_bridges)
            )
        ),
    }
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
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
        },
        "phase_output_relative_l2": phase_output_relative_l2,
        "forward": {
            "mapped_rht_ms": reference_forward_ms,
            "fused_adamw_rht_ms": fused_forward_ms,
            "speedup": reference_forward_ms / fused_forward_ms,
        },
        "backward": {
            "mapped_rht_ms": reference_backward_ms,
            "fused_adamw_rht_ms": fused_backward_ms,
            "speedup": reference_backward_ms / fused_backward_ms,
        },
        "reference_step_ms": reference_ms,
        "fused_step_ms": fused_ms,
        "full_step_speedup": reference_ms / fused_ms,
        "reference_optimizer_tail_ms": reference_tail_ms,
        "fused_optimizer_tail_ms": fused_tail_ms,
        "optimizer_tail_speedup": reference_tail_ms / fused_tail_ms,
        "one_step_correctness": one_step_correctness,
        "post_full_timing_drift": post_full_timing_drift,
        "post_tail_timing_drift": post_tail_timing_drift,
        "samples_ms": {
            "mapped_rht_forward": forward_samples[0],
            "fused_adamw_rht_forward": forward_samples[1],
            "mapped_rht_backward": backward_samples[0],
            "fused_adamw_rht_backward": backward_samples[1],
            "reference_step": reference_samples,
            "fused_step": fused_samples,
            "reference_optimizer_tail": reference_tail,
            "fused_optimizer_tail": fused_tail,
        },
        "scope_note": (
            "Algorithm-matched comparison: both paths use identical RHT math, "
            "direct TE 2D block-FP8 weights, forward, backward, FP32 masters, "
            "and AdamW hyperparameters. Reference maps BF16 working Wgrad "
            "through R^T into an FP32 tensor, then calls foreach PyTorch AdamW. "
            "Fused consumes the same rotated Wgrad and updates original-basis "
            "FP32 masters/moments directly."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "reference_step_ms": reference_ms,
        "fused_step_ms": fused_ms,
        "full_step_speedup": payload["full_step_speedup"],
        "reference_optimizer_tail_ms": reference_tail_ms,
        "fused_optimizer_tail_ms": fused_tail_ms,
        "optimizer_tail_speedup": payload["optimizer_tail_speedup"],
        "phase_output_relative_l2": phase_output_relative_l2,
        "forward": payload["forward"],
        "backward": payload["backward"],
        **{f"one_step_{key}": value for key, value in one_step_correctness.items()},
        "all_finite": post_tail_timing_drift["all_finite"],
    }, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
