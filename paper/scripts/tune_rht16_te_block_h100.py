#!/usr/bin/env python3
"""Sweep launch configurations for the fused H16/TE block-FP8 writers."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path("paper/rht").resolve()))
from rht16_te_block import (  # noqa: E402
    _rht16_te_block_both_kernel,
    _rht16_te_block_columnwise_kernel,
    _rht16_te_block_kernel,
)
from rht16_triton import DEFAULT_SIGN_MASK  # noqa: E402


def measure(fn, warmup: int = 20, iterations: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=11008)
    args = parser.parse_args()
    rows, width = args.rows, args.width
    if rows % 128 or width % 128:
        raise ValueError("rows and width must be divisible by 128")

    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((width // 128, rows), device="cuda", dtype=torch.float32)
    q_col = torch.empty((width, rows), device="cuda", dtype=torch.float8_e4m3fn)
    col_scale = torch.empty((rows // 128, width), device="cuda", dtype=torch.float32)

    print(f"GPU: {torch.cuda.get_device_name(0)}; shape=({rows}, {width})")
    row_outputs = []
    for block_m in (16, 64):
        _rht16_te_block_kernel[(triton.cdiv(rows, block_m), width // 128)](
            x,
            q,
            row_scale,
            rows=rows,
            padded_rows=rows,
            BLOCK_M=block_m,
            SIGN_MASK=DEFAULT_SIGN_MASK,
            SWIGLU=False,
            num_warps=4,
        )
        row_outputs.append((q.view(torch.uint8).clone(), row_scale.clone()))
    print(
        "block64-vs-block16: "
        f"byte_agreement={(row_outputs[0][0] == row_outputs[1][0]).float().mean():.9f} "
        f"scale_max_error={(row_outputs[0][1] - row_outputs[1][1]).abs().max():.9f}"
    )
    print("rowwise")
    for block_m in (16, 32, 64, 128):
        for num_warps in (4, 8):
            for num_stages in (1, 2, 3):
                def launch_row(
                    block_m: int = block_m,
                    num_warps: int = num_warps,
                    num_stages: int = num_stages,
                ) -> None:
                    _rht16_te_block_kernel[
                        (triton.cdiv(rows, block_m), width // 128)
                    ](
                        x,
                        q,
                        row_scale,
                        rows=rows,
                        padded_rows=rows,
                        BLOCK_M=block_m,
                        SIGN_MASK=DEFAULT_SIGN_MASK,
                        SWIGLU=False,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )

                try:
                    elapsed = measure(launch_row)
                    print(
                        f"  block_m={block_m:2d} warps={num_warps} "
                        f"stages={num_stages}: {elapsed:.6f} ms",
                        flush=True,
                    )
                except Exception as error:
                    print(
                        f"  block_m={block_m:2d} warps={num_warps} "
                        f"stages={num_stages}: ERROR {error}",
                        flush=True,
                    )

    print("columnwise")
    for num_warps in (4, 8):
        for num_stages in (1, 2, 3):
            def launch_col(
                num_warps: int = num_warps,
                num_stages: int = num_stages,
            ) -> None:
                _rht16_te_block_columnwise_kernel[(rows // 128, width // 16)](
                    x,
                    q_col,
                    col_scale,
                    width=width,
                    rows=rows,
                    padded_width=width,
                    SIGN_MASK=DEFAULT_SIGN_MASK,
                    SWIGLU=False,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            elapsed = measure(launch_col)
            print(
                f"  warps={num_warps} stages={num_stages}: {elapsed:.6f} ms",
                flush=True,
            )

    print("combined rowwise+columnwise")
    for num_warps in (4, 8):
        for num_stages in (1, 2, 3):
            def launch_both(
                num_warps: int = num_warps,
                num_stages: int = num_stages,
            ) -> None:
                _rht16_te_block_both_kernel[(rows // 128, width // 128)](
                    x,
                    q,
                    row_scale,
                    q_col,
                    col_scale,
                    rows=rows,
                    padded_rows=rows,
                    padded_width=width,
                    SIGN_MASK=DEFAULT_SIGN_MASK,
                    SWIGLU=False,
                    INPUT_FP32=False,
                    SCALE_2D=False,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            elapsed = measure(launch_both)
            print(
                f"  warps={num_warps} stages={num_stages}: {elapsed:.6f} ms",
                flush=True,
            )

    x_swiglu = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16)
    print("combined SwiGLU+rowwise+columnwise")
    for num_warps in (4, 8):
        for num_stages in (1, 2, 3):
            def launch_swiglu_both(
                num_warps: int = num_warps,
                num_stages: int = num_stages,
            ) -> None:
                _rht16_te_block_both_kernel[(rows // 128, width // 128)](
                    x_swiglu,
                    q,
                    row_scale,
                    q_col,
                    col_scale,
                    rows=rows,
                    padded_rows=rows,
                    padded_width=width,
                    SIGN_MASK=DEFAULT_SIGN_MASK,
                    SWIGLU=True,
                    INPUT_FP32=False,
                    SCALE_2D=False,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            elapsed = measure(launch_swiglu_both)
            print(
                f"  warps={num_warps} stages={num_stages}: {elapsed:.6f} ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
