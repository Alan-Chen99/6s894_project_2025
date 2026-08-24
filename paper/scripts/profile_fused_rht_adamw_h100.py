#!/usr/bin/env python3
"""Emit NVTX ranges for mapped versus fused inverse-RHT AdamW profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--warmup", type=int, default=4)
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

    mapped_fc1, mapped_down, mapped_bridges = make_path()
    fused_fc1, fused_down, fused_bridges = make_path()
    mapped_optimizer = torch.optim.AdamW(
        [bridge.master for bridge in mapped_bridges],
        lr=1.0e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=True,
    )
    fused_optimizer = FusedRHTAdamW(
        fused_bridges,
        lr=1.0e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    mapped_x = torch.randn(
        m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    fused_x = mapped_x.detach().clone().requires_grad_(True)
    dy = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    def mlp(x, fc1, down):
        with te.autocast(enabled=True, recipe=block_recipe):
            z = fc1(rht16_te_block_autograd(x))
            hidden = swiglu_rht16_te_block_autograd(z)
            return down(hidden)

    def mapped_step() -> None:
        mapped_x.grad = None
        mapped_optimizer.zero_grad(set_to_none=True)
        clear_all_working_grads(mapped_bridges)
        materialize_all(mapped_bridges)
        mlp(mapped_x, mapped_fc1, mapped_down).backward(dy)
        map_all_grads(mapped_bridges)
        mapped_optimizer.step()

    def fused_step() -> None:
        fused_x.grad = None
        fused_optimizer.zero_grad(set_to_none=True)
        materialize_all(fused_bridges)
        mlp(fused_x, fused_fc1, fused_down).backward(dy)
        fused_optimizer.step()

    for _ in range(args.warmup):
        mapped_step()
        fused_step()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("profile_window")
    torch.cuda.nvtx.range_push("mapped_full_step")
    mapped_step()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("fused_full_step")
    fused_step()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("mapped_optimizer_tail")
    mapped_optimizer.zero_grad(set_to_none=True)
    map_all_grads(mapped_bridges)
    mapped_optimizer.step()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("fused_optimizer_tail")
    fused_optimizer.step()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
