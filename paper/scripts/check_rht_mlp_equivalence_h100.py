#!/usr/bin/env python3
"""Check paired activation/weight RHT equivalence for a SwiGLU MLP."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.ops.basic import SwiGLU

sys.path.insert(0, str(Path("paper/rht").resolve()))
from rht16_te_block import (  # noqa: E402
    rht16_te_block_autograd,
    swiglu_rht16_te_block_autograd,
)
from rht16_triton import (  # noqa: E402
    reference_matrix,
    rht16,
    rht16_autograd,
    rht16_transpose,
)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().float()
    expected = expected.detach().float()
    return float((actual - expected).norm() / expected.norm())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m, k, n = args.tokens, args.hidden, args.intermediate
    if any(value % 16 for value in (k, n)):
        raise ValueError("hidden and intermediate must be divisible by 16")

    torch.manual_seed(1234)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = (
        torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    ).requires_grad_(True)
    w2 = (
        torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / math.sqrt(n)
    ).requires_grad_(True)
    dy = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    z = x @ w1.T
    gate, up = z.chunk(2, dim=-1)
    y = (F.silu(gate) * up) @ w2.T
    (y.float() * dy.float()).sum().backward()
    reference = {
        "output": y.detach(),
        "x_grad": x.grad.detach().clone(),
        "w1_grad": w1.grad.detach().clone(),
        "w2_grad": w2.grad.detach().clone(),
    }

    xr = x.detach().clone().requires_grad_(True)
    w1r_source = w1.detach().clone().requires_grad_(True)
    w2r_source = w2.detach().clone().requires_grad_(True)
    xr_value = rht16_autograd(xr.reshape(-1, 16)).reshape_as(xr)
    w1r = rht16_autograd(w1r_source.reshape(-1, 16)).reshape_as(w1r_source)
    zr = xr_value @ w1r.T
    gate_r, up_r = zr.chunk(2, dim=-1)
    hidden_r = F.silu(gate_r) * up_r
    hidden_rotated = rht16_autograd(hidden_r.reshape(-1, 16)).reshape_as(hidden_r)
    w2r = rht16_autograd(w2r_source.reshape(-1, 16)).reshape_as(w2r_source)
    yr = hidden_rotated @ w2r.T
    (yr.float() * dy.float()).sum().backward()

    matrix = reference_matrix(device=x.device, dtype=torch.float32)

    def dense_rht(value: torch.Tensor) -> torch.Tensor:
        return (value.reshape(-1, 16) @ matrix).reshape_as(value)

    x32 = x.detach().float().requires_grad_(True)
    w1_32 = w1.detach().float().requires_grad_(True)
    w2_32 = w2.detach().float().requires_grad_(True)
    z32 = x32 @ w1_32.T
    gate32, up32 = z32.chunk(2, dim=-1)
    y32 = (F.silu(gate32) * up32) @ w2_32.T
    (y32 * dy.float()).sum().backward()

    x32r = x.detach().float().requires_grad_(True)
    w1_32r = w1.detach().float().requires_grad_(True)
    w2_32r = w2.detach().float().requires_grad_(True)
    z32r = dense_rht(x32r) @ dense_rht(w1_32r).T
    gate32r, up32r = z32r.chunk(2, dim=-1)
    hidden32r = F.silu(gate32r) * up32r
    y32r = dense_rht(hidden32r) @ dense_rht(w2_32r).T
    (y32r * dy.float()).sum().backward()

    x_te = x.detach().clone().requires_grad_(True)
    fc1_te = te.Linear(k, 2 * n, bias=False, params_dtype=torch.bfloat16, device="cuda")
    fc2_te = te.Linear(n, k, bias=False, params_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        fc1_te.weight.copy_(rht16(w1.detach().reshape(-1, 16)).reshape_as(w1))
        fc2_te.weight.copy_(rht16(w2.detach().reshape(-1, 16)).reshape_as(w2))
    block_recipe = recipe.Float8BlockScaling()
    with te.autocast(enabled=True, recipe=block_recipe):
        z_te = fc1_te(rht16_te_block_autograd(x_te))
        y_te = fc2_te(swiglu_rht16_te_block_autograd(z_te))
    (y_te.float() * dy.float()).sum().backward()
    fc1_grad_original_basis = rht16_transpose(
        fc1_te.weight.grad.contiguous().reshape(-1, 16)
    ).reshape_as(fc1_te.weight.grad)
    fc2_grad_original_basis = rht16_transpose(
        fc2_te.weight.grad.contiguous().reshape(-1, 16)
    ).reshape_as(fc2_te.weight.grad)

    x_plain = x.detach().clone().requires_grad_(True)
    fc1_plain = te.Linear(k, 2 * n, bias=False, params_dtype=torch.bfloat16, device="cuda")
    fc2_plain = te.Linear(n, k, bias=False, params_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        fc1_plain.weight.copy_(w1.detach())
        fc2_plain.weight.copy_(w2.detach())
    with te.autocast(enabled=True, recipe=block_recipe):
        z_plain = fc1_plain(x_plain)
        hidden_plain = SwiGLU()(z_plain)
        y_plain = fc2_plain(hidden_plain)
    (y_plain.float() * dy.float()).sum().backward()

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "shape": {"tokens": m, "hidden": k, "intermediate": n},
        "identity": "(xR)(wR)^T = xw^T applied at FC1 and FC2 input axes",
        "output_relative_l2": relative_l2(yr, reference["output"]),
        "input_grad_relative_l2": relative_l2(xr.grad, reference["x_grad"]),
        "fc1_weight_grad_relative_l2": relative_l2(
            w1r_source.grad, reference["w1_grad"]
        ),
        "fc2_weight_grad_relative_l2": relative_l2(
            w2r_source.grad, reference["w2_grad"]
        ),
        "output_max_abs_error": float(
            (yr.detach() - reference["output"]).abs().max()
        ),
        "float32_identity": {
            "output_relative_l2": relative_l2(y32r, y32),
            "input_grad_relative_l2": relative_l2(x32r.grad, x32.grad),
            "fc1_weight_grad_relative_l2": relative_l2(w1_32r.grad, w1_32.grad),
            "fc2_weight_grad_relative_l2": relative_l2(w2_32r.grad, w2_32.grad),
        },
        "te_block_fp8_paired": {
            "output_relative_l2_vs_bf16": relative_l2(y_te, reference["output"]),
            "input_grad_relative_l2_vs_bf16": relative_l2(
                x_te.grad, reference["x_grad"]
            ),
            "fc1_weight_grad_relative_l2_vs_bf16": relative_l2(
                fc1_grad_original_basis, reference["w1_grad"]
            ),
            "fc2_weight_grad_relative_l2_vs_bf16": relative_l2(
                fc2_grad_original_basis, reference["w2_grad"]
            ),
            "all_finite": bool(
                torch.isfinite(y_te).all()
                and torch.isfinite(x_te.grad).all()
                and torch.isfinite(fc1_te.weight.grad).all()
                and torch.isfinite(fc2_te.weight.grad).all()
            ),
        },
        "te_block_fp8_plain": {
            "output_relative_l2_vs_bf16": relative_l2(y_plain, reference["output"]),
            "input_grad_relative_l2_vs_bf16": relative_l2(
                x_plain.grad, reference["x_grad"]
            ),
            "fc1_weight_grad_relative_l2_vs_bf16": relative_l2(
                fc1_plain.weight.grad, reference["w1_grad"]
            ),
            "fc2_weight_grad_relative_l2_vs_bf16": relative_l2(
                fc2_plain.weight.grad, reference["w2_grad"]
            ),
        },
        "all_finite": bool(
            torch.isfinite(yr).all()
            and torch.isfinite(xr.grad).all()
            and torch.isfinite(w1r_source.grad).all()
            and torch.isfinite(w2r_source.grad).all()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
