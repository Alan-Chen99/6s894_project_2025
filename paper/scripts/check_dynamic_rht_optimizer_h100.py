#!/usr/bin/env python3
"""Validate dynamic RHT materialization and original-basis Adam updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("paper/rht").resolve()))
from dynamic_weight import DynamicWeightBridge  # noqa: E402
from rht16_triton import reference_matrix  # noqa: E402


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().float()
    expected = expected.detach().float()
    return float((actual - expected).norm() / expected.norm())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.width % 16:
        raise ValueError("width must be divisible by 16")

    torch.manual_seed(1234)
    initial = torch.randn(args.rows, args.width, device="cuda", dtype=torch.float32)
    working = torch.nn.Parameter(torch.empty_like(initial, dtype=torch.bfloat16))
    bridge = DynamicWeightBridge.from_working_weight(
        working, rotated=True, initial=initial
    )
    matrix = reference_matrix(device=initial.device, dtype=torch.float32)

    bridge.materialize()
    expected_working = (
        initial.reshape(-1, 16).to(torch.bfloat16) @ matrix.to(torch.bfloat16)
    ).reshape_as(working)
    materialize_error = relative_l2(working, expected_working)

    working.grad = torch.randn_like(working)
    expected_master_grad = (
        working.grad.reshape(-1, 16).float() @ matrix.T
    ).reshape_as(initial)
    bridge.map_grad_to_master()
    gradient_error = relative_l2(bridge.master.grad, expected_master_grad)

    reference_master = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([bridge.master], lr=1.0e-3, weight_decay=0.01)
    reference_optimizer = torch.optim.AdamW(
        [reference_master], lr=1.0e-3, weight_decay=0.01
    )
    reference_master.grad = bridge.master.grad.detach().clone()
    optimizer.step()
    reference_optimizer.step()
    update_error = relative_l2(bridge.master, reference_master)
    bridge.materialize()
    expected_after_update = (
        bridge.master.detach().reshape(-1, 16).to(torch.bfloat16)
        @ matrix.to(torch.bfloat16)
    ).reshape_as(working)
    rematerialize_error = relative_l2(working, expected_after_update)

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "shape": [args.rows, args.width],
        "master_dtype": str(bridge.master.dtype),
        "working_dtype": str(working.dtype),
        "identity": "W_work=WR; grad_master=grad_work R^T",
        "materialize_relative_l2": materialize_error,
        "gradient_map_relative_l2": gradient_error,
        "adam_update_relative_l2": update_error,
        "rematerialize_relative_l2": rematerialize_error,
        "all_finite": bool(
            torch.isfinite(working).all()
            and torch.isfinite(bridge.master).all()
            and torch.isfinite(bridge.master.grad).all()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
