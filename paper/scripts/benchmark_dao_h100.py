"""Matched saturated-throughput comparison with Tri Dao's CUDA FWHT."""

import json
import math
from pathlib import Path

import torch

import csrc
from fast_hadamard_transform import hadamard_transform as dao_hadamard_transform


SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
DTYPES = (torch.float16, torch.bfloat16)
TOTAL_ELEMENTS = 1 << 30
WARMUP = 10
ITERS = 400


def time_many(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / ITERS


def main() -> None:
    torch.manual_seed(0)
    rows = []
    for dtype in DTYPES:
        for size in SIZES:
            batch = TOTAL_ELEMENTS // size
            source = torch.randn(batch, size, device="cuda", dtype=dtype)
            ours_input = source.clone()
            dao_input = source.clone()
            scale = 1.0 / math.sqrt(size)

            ours_ms = time_many(
                lambda: csrc.hadamard_transform(ours_input, inplace=True)
            )
            dao_ms = time_many(lambda: dao_hadamard_transform(dao_input, scale=scale))
            row = {
                "dtype": str(dtype).removeprefix("torch."),
                "size": size,
                "batch": batch,
                "total_elements": TOTAL_ELEMENTS,
                "iterations": ITERS,
                "ours_ms": ours_ms,
                "dao_ms": dao_ms,
                "speedup_vs_dao": dao_ms / ours_ms,
            }
            rows.append(row)
            print(row, flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "dao_commit": "e7706faf8d1c3b9f241e36860640ad1dac644ede",
        "rows": rows,
    }
    output = Path("paper/results/raw/h100_sm90_dao_comparison_2026-08-23.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
