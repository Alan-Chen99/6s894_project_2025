#!/usr/bin/env python3
"""Capture one warmed TE RHT forward or training step for Nsight Systems."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

sys.path.insert(0, str(Path("paper/rht").resolve()))
from rht16_te_block import rht16_te_block_autograd  # noqa: E402
from rht16_triton import rht16_autograd  # noqa: E402


SHAPES = {
    "square_4096": (4096, 4096, 4096),
    "square_8192": (4096, 8192, 8192),
    "llama_up": (4096, 4096, 11008),
    "llama_down": (4096, 11008, 4096),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=SHAPES, required=True)
    parser.add_argument("--method", choices=("separate", "fused"), required=True)
    parser.add_argument("--phase", choices=("forward", "train"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    m, k, n = SHAPES[args.shape]
    torch.manual_seed(1234)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    layer = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
    block_recipe = recipe.Float8BlockScaling()

    make_input = (
        (lambda: rht16_autograd(x))
        if args.method == "separate"
        else (lambda: rht16_te_block_autograd(x))
    )

    def step() -> torch.Tensor:
        layer.zero_grad(set_to_none=True)
        x.grad = None
        q = make_input()
        with te.autocast(enabled=True, recipe=block_recipe):
            y = layer(q)
        if args.phase == "train":
            y.backward(dy)
        return y

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    label = f"{args.method}_{args.phase}_{args.shape}"
    torch.cuda.nvtx.range_push(label)
    step()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
