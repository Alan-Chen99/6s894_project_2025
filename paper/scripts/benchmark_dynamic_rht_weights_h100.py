#!/usr/bin/env python3
"""Measure dynamic original-basis RHT weight overhead in a TE FP8 MLP."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

sys.path.insert(0, str(Path("paper/rht").resolve()))
from dynamic_weight import (  # noqa: E402
    DynamicQuantizedWeightBridge,
    DynamicWeightBridge,
    clear_all_working_grads,
    map_all_grads,
    materialize_all,
)
from rht16_te_block import (  # noqa: E402
    rht16_te_block_autograd,
    swiglu_rht16_te_block_autograd,
)
from rht16_triton import rht16, rht16_into  # noqa: E402


def measure_paired(fn_a, fn_b, warmup: int, iterations: int):
    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    samples = [[], []]
    rng = random.Random(1234)
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


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().float()
    expected = expected.detach().float()
    return float((actual - expected).norm() / expected.norm())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-method", choices=("static", "dynamic", "fused"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m, k, n = args.tokens, args.hidden, args.intermediate
    if any(value % 128 for value in (m, k, n)):
        raise ValueError("all dimensions must be divisible by 128")

    torch.manual_seed(1234)
    initial_w1 = torch.randn(2 * n, k, device="cuda") / (k**0.5)
    initial_w2 = torch.randn(k, n, device="cuda") / (n**0.5)

    block_recipe = recipe.Float8BlockScaling()

    def make_layers(*, quantized: bool = False):
        context = (
            te.quantized_model_init(enabled=True, recipe=block_recipe)
            if quantized
            else nullcontext()
        )
        with context:
            return (
                te.Linear(k, 2 * n, bias=False, params_dtype=torch.bfloat16, device="cuda"),
                te.Linear(n, k, bias=False, params_dtype=torch.bfloat16, device="cuda"),
            )

    static_fc1, static_down = make_layers()
    dynamic_fc1, dynamic_down = make_layers()
    fused_fc1, fused_down = make_layers(quantized=True)
    with torch.no_grad():
        rht16_into(initial_w1.reshape(-1, 16), static_fc1.weight.reshape(-1, 16))
        rht16_into(initial_w2.reshape(-1, 16), static_down.weight.reshape(-1, 16))
    bridges = [
        DynamicWeightBridge.from_working_weight(
            dynamic_fc1.weight, rotated=True, initial=initial_w1
        ),
        DynamicWeightBridge.from_working_weight(
            dynamic_down.weight, rotated=True, initial=initial_w2
        ),
    ]
    fused_bridges = [
        DynamicQuantizedWeightBridge.attach(fused_fc1, initial_w1),
        DynamicQuantizedWeightBridge.attach(fused_down, initial_w2),
    ]
    materialize_all(bridges)
    materialize_all(fused_bridges)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    def mlp(fc1, down):
        with te.autocast(enabled=True, recipe=block_recipe):
            z = fc1(rht16_te_block_autograd(x))
            hidden = swiglu_rht16_te_block_autograd(z)
            return down(hidden)

    def clear_static():
        x.grad = None
        static_fc1.zero_grad(set_to_none=True)
        static_down.zero_grad(set_to_none=True)

    def static_step():
        clear_static()
        y = mlp(static_fc1, static_down)
        y.backward(dy)
        return y

    def dynamic_step():
        x.grad = None
        clear_all_working_grads(bridges)
        for bridge in bridges:
            bridge.master.grad = None
        materialize_all(bridges)
        y = mlp(dynamic_fc1, dynamic_down)
        y.backward(dy)
        map_all_grads(bridges)
        return y

    def fused_dynamic_step():
        x.grad = None
        clear_all_working_grads(fused_bridges)
        for bridge in fused_bridges:
            bridge.master.grad = None
        materialize_all(fused_bridges)
        y = mlp(fused_fc1, fused_down)
        y.backward(dy)
        map_all_grads(fused_bridges)
        return y

    if args.profile_method is not None:
        selected = {
            "static": static_step,
            "dynamic": dynamic_step,
            "fused": fused_dynamic_step,
        }[args.profile_method]
        for _ in range(args.warmup):
            selected()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(
            f"dynamic_weight_{args.profile_method}_{m}_{k}_{n}"
        )
        selected()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    static_samples, dynamic_samples = measure_paired(
        static_step, dynamic_step, args.warmup, args.iterations
    )
    static_for_fused_samples, fused_samples = measure_paired(
        static_step, fused_dynamic_step, args.warmup, args.iterations
    )
    matched_dynamic_samples, matched_fused_samples = measure_paired(
        dynamic_step, fused_dynamic_step, args.warmup, args.iterations
    )

    eager_buffers = [torch.empty_like(bridge.working) for bridge in bridges]

    @torch.no_grad()
    def eager_materialize():
        for bridge, out in zip(bridges, eager_buffers):
            source = bridge.master.to(bridge.working.dtype)
            out.copy_(rht16(source.reshape(-1, 16)).reshape_as(out))

    direct_materialize, eager_materialize_samples = measure_paired(
        lambda: materialize_all(bridges),
        eager_materialize,
        args.warmup,
        args.iterations,
    )
    _, fused_2d_materialize_samples = measure_paired(
        lambda: materialize_all(bridges),
        lambda: materialize_all(fused_bridges),
        args.warmup,
        args.iterations,
    )

    # Populate representative gradients before timing the map itself.
    dynamic_step()
    map_samples, identity_copy_samples = measure_paired(
        lambda: map_all_grads(bridges),
        lambda: [
            bridge.master.grad.copy_(bridge.working.grad) for bridge in bridges
        ],
        args.warmup,
        args.iterations,
    )

    clear_static()
    static_y = mlp(static_fc1, static_down)
    static_y.backward(dy)
    dynamic_y = dynamic_step()
    fused_y = fused_dynamic_step()
    reference_weight_quantizer = te.Float8BlockQuantizer(
        te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=2,
    )
    reference_qweight = reference_weight_quantizer(static_fc1.weight)

    medians = {
        "static_train_ms": statistics.median(static_samples),
        "dynamic_train_ms": statistics.median(dynamic_samples),
        "fused_quantized_dynamic_train_ms": statistics.median(fused_samples),
        "matched_dynamic_train_ms": statistics.median(matched_dynamic_samples),
        "matched_fused_quantized_dynamic_train_ms": statistics.median(
            matched_fused_samples
        ),
        "direct_materialize_ms": statistics.median(direct_materialize),
        "eager_materialize_ms": statistics.median(eager_materialize_samples),
        "fused_2d_fp8_materialize_ms": statistics.median(
            fused_2d_materialize_samples
        ),
        "gradient_map_ms": statistics.median(map_samples),
        "identity_gradient_copy_ms": statistics.median(identity_copy_samples),
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
        "optimizer_master": "FP32, original basis",
        **medians,
        "dynamic_vs_static_ratio": medians["dynamic_train_ms"]
        / medians["static_train_ms"],
        "fused_quantized_dynamic_vs_static_ratio": medians[
            "fused_quantized_dynamic_train_ms"
        ]
        / medians["static_train_ms"],
        "fused_quantized_speedup_vs_bf16_working_dynamic": medians[
            "matched_dynamic_train_ms"
        ] / medians["matched_fused_quantized_dynamic_train_ms"],
        "direct_materialize_speedup_vs_eager": medians["eager_materialize_ms"]
        / medians["direct_materialize_ms"],
        "correctness": {
            "output_relative_l2": relative_l2(dynamic_y, static_y),
            "fused_weight_output_relative_l2_vs_static": relative_l2(
                fused_y, static_y
            ),
            "fused_vs_bf16_working_fc1_master_grad_relative_l2": relative_l2(
                fused_bridges[0].master.grad, bridges[0].master.grad
            ),
            "fused_vs_bf16_working_down_master_grad_relative_l2": relative_l2(
                fused_bridges[1].master.grad, bridges[1].master.grad
            ),
            "fc1_rowwise_bytes_match_te_2d_reference": bool(
                torch.equal(
                    fused_fc1.weight._rowwise_data,
                    reference_qweight._rowwise_data,
                )
            ),
            "fc1_rowwise_scales_match_te_2d_reference": bool(
                torch.equal(
                    fused_fc1.weight._rowwise_scale_inv[:, : k // 128],
                    reference_qweight._rowwise_scale_inv[:, : k // 128],
                )
            ),
            "fc1_columnwise_bytes_match_te_2d_reference": bool(
                torch.equal(
                    fused_fc1.weight._columnwise_data,
                    reference_qweight._columnwise_data,
                )
            ),
            "fc1_columnwise_scales_match_te_2d_reference": bool(
                torch.equal(
                    fused_fc1.weight._columnwise_scale_inv[:, : (2 * n) // 128],
                    reference_qweight._columnwise_scale_inv[:, : (2 * n) // 128],
                )
            ),
            "fc1_rowwise_byte_match_fraction": float(
                (
                    fused_fc1.weight._rowwise_data
                    == reference_qweight._rowwise_data
                ).float().mean()
            ),
            "fc1_columnwise_byte_match_fraction": float(
                (
                    fused_fc1.weight._columnwise_data
                    == reference_qweight._columnwise_data
                ).float().mean()
            ),
            "fc1_row_scale_max_abs_difference": float(
                (
                    fused_fc1.weight._rowwise_scale_inv[:, : k // 128]
                    - reference_qweight._rowwise_scale_inv[:, : k // 128]
                ).abs().max()
            ),
            "fc1_col_scale_max_abs_difference": float(
                (
                    fused_fc1.weight._columnwise_scale_inv[:, : (2 * n) // 128]
                    - reference_qweight._columnwise_scale_inv[:, : (2 * n) // 128]
                ).abs().max()
            ),
            "all_finite": bool(
                torch.isfinite(dynamic_y).all()
                and torch.isfinite(fused_y).all()
                and all(
                    torch.isfinite(bridge.master.grad).all()
                    for bridge in (*bridges, *fused_bridges)
                )
            ),
        },
        "samples_ms": {
            "static_train": static_samples,
            "dynamic_train": dynamic_samples,
            "static_train_paired_with_fused": static_for_fused_samples,
            "fused_quantized_dynamic_train": fused_samples,
            "matched_dynamic_train": matched_dynamic_samples,
            "matched_fused_quantized_dynamic_train": matched_fused_samples,
            "direct_materialize": direct_materialize,
            "eager_materialize": eager_materialize_samples,
            "fused_2d_fp8_materialize": fused_2d_materialize_samples,
            "gradient_map": map_samples,
            "identity_gradient_copy": identity_copy_samples,
        },
        "scope_note": (
            "Dynamic steps include FP32-master WR materialization and R^T Wgrad "
            "mapping, but exclude AdamW. The fused path writes TE-compatible 1D "
            "block-FP8 weight buffers directly and skips TE weight quantization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({**medians, **payload["correctness"]}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
