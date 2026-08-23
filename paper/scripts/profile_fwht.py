"""Emit one NVTX-marked FWHT launch for Nsight Compute/Systems."""

import argparse

import torch

import csrc
import hada_core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("ours", "hadacore"), required=True)
    parser.add_argument("--size", type=int, default=32768)
    parser.add_argument("--elements", type=int, default=1 << 26)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if args.elements % args.size:
        parser.error("--elements must be divisible by --size")
    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.elements // args.size, args.size, device="cuda", dtype=dtype)
    fn = csrc.hadamard_transform if args.impl == "ours" else hada_core.hadamard_transform

    for _ in range(args.warmup):
        fn(x, inplace=True)
    torch.cuda.synchronize()

    range_name = f"profile_{args.impl}_{args.dtype}_n{args.size}"
    torch.cuda.nvtx.range_push(range_name)
    fn(x, inplace=True)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
