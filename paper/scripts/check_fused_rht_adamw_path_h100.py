#!/usr/bin/env python3
"""Check path-level equivalence of mapped and fused RHT AdamW updates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("paper/scripts").resolve()))
from train_te_rht_convergence_h100 import TrainingPath  # noqa: E402


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

    torch.manual_seed(2027)
    teacher_w1 = torch.randn(2 * n, k, device="cuda") / math.sqrt(k)
    teacher_w2 = torch.randn(k, n, device="cuda") / math.sqrt(n)
    initial_w1 = teacher_w1 + 0.08 * torch.randn_like(teacher_w1) / math.sqrt(k)
    initial_w2 = teacher_w2 + 0.08 * torch.randn_like(teacher_w2) / math.sqrt(n)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    target = torch.randn_like(x)
    mapped = TrainingPath(
        "mapped",
        initial_w1,
        initial_w2,
        fp8=True,
        rotated=True,
        fused_rht_adamw=False,
        learning_rate=1.0e-4,
    )
    fused = TrainingPath(
        "fused",
        initial_w1,
        initial_w2,
        fp8=True,
        rotated=True,
        fused_rht_adamw=True,
        learning_rate=1.0e-4,
    )

    mapped.optimizer.zero_grad(set_to_none=True)
    fused.optimizer.zero_grad(set_to_none=True)
    mapped_x = x.detach().requires_grad_(True)
    fused_x = x.detach().requires_grad_(True)
    mapped_y = mapped.forward(mapped_x)
    fused_y = fused.forward(fused_x)
    mapped_loss = F.mse_loss(mapped_y.float(), target.float())
    fused_loss = F.mse_loss(fused_y.float(), target.float())
    mapped_loss.backward()
    fused_loss.backward()

    before_update = {
        "fc1_master_storage_independent": (
            mapped.bridges[0].master.data_ptr() != fused.bridges[0].master.data_ptr()
        ),
        "down_master_storage_independent": (
            mapped.bridges[1].master.data_ptr() != fused.bridges[1].master.data_ptr()
        ),
        "fc1_row_bytes_equal": bool(torch.equal(
            mapped.fc1.weight._rowwise_data, fused.fc1.weight._rowwise_data
        )),
        "fc1_col_bytes_equal": bool(torch.equal(
            mapped.fc1.weight._columnwise_data, fused.fc1.weight._columnwise_data
        )),
        "down_row_bytes_equal": bool(torch.equal(
            mapped.down.weight._rowwise_data, fused.down.weight._rowwise_data
        )),
        "down_col_bytes_equal": bool(torch.equal(
            mapped.down.weight._columnwise_data, fused.down.weight._columnwise_data
        )),
        "output_relative_l2": relative_l2(fused_y, mapped_y),
        "loss_absolute_difference": abs(
            float(fused_loss.detach()) - float(mapped_loss.detach())
        ),
        "input_grad_relative_l2": relative_l2(fused_x.grad, mapped_x.grad),
        "fc1_working_grad_relative_l2": relative_l2(
            fused.fc1.weight.grad, mapped.fc1.weight.grad
        ),
        "down_working_grad_relative_l2": relative_l2(
            fused.down.weight.grad, mapped.down.weight.grad
        ),
    }
    from dynamic_weight import map_all_grads  # noqa: E402

    map_all_grads(mapped.bridges)
    mapped.optimizer.step()
    fused.optimizer.step()
    mapped_fc1_state = mapped.optimizer.state[mapped.bridges[0].master]
    mapped_down_state = mapped.optimizer.state[mapped.bridges[1].master]
    after_update = {
        "fc1_master_relative_l2": relative_l2(
            fused.bridges[0].master, mapped.bridges[0].master
        ),
        "down_master_relative_l2": relative_l2(
            fused.bridges[1].master, mapped.bridges[1].master
        ),
        "fc1_exp_avg_relative_l2": relative_l2(
            fused.optimizer.state[0]["exp_avg"], mapped_fc1_state["exp_avg"]
        ),
        "fc1_exp_avg_sq_relative_l2": relative_l2(
            fused.optimizer.state[0]["exp_avg_sq"], mapped_fc1_state["exp_avg_sq"]
        ),
        "down_exp_avg_relative_l2": relative_l2(
            fused.optimizer.state[1]["exp_avg"], mapped_down_state["exp_avg"]
        ),
        "down_exp_avg_sq_relative_l2": relative_l2(
            fused.optimizer.state[1]["exp_avg_sq"], mapped_down_state["exp_avg_sq"]
        ),
    }
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "shape": {"tokens": m, "hidden": k, "intermediate": n},
        "before_update": before_update,
        "after_update": after_update,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
