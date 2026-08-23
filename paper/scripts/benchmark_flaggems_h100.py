"""Matched saturated-throughput comparison with FlagGems' Triton FWHT."""

import importlib.util
import json
import math
from pathlib import Path

import torch

import csrc


SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
DTYPES = (torch.float16, torch.bfloat16)
TOTAL_ELEMENTS = 1 << 30
WARMUP = 10
ITERS = 400


def load_flaggems_operator():
    source = Path(
        "paper/baselines/src/FlagGems/src/flag_gems/ops/hadamard_transform.py"
    )
    spec = importlib.util.spec_from_file_location("flaggems_hadamard", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.hadamard_transform


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
    flaggems_hadamard_transform = load_flaggems_operator()
    torch.manual_seed(0)
    rows = []
    for dtype in DTYPES:
        for size in SIZES:
            batch = TOTAL_ELEMENTS // size
            source = torch.randn(batch, size, device="cuda", dtype=dtype)
            ours_input = source.clone()
            flaggems_input = source.clone()
            scale = 1.0 / math.sqrt(size)

            ours_ms = time_many(
                lambda: csrc.hadamard_transform(ours_input, inplace=True)
            )
            flaggems_ms = time_many(
                lambda: flaggems_hadamard_transform(flaggems_input, scale=scale)
            )
            row = {
                "dtype": str(dtype).removeprefix("torch."),
                "size": size,
                "batch": batch,
                "total_elements": TOTAL_ELEMENTS,
                "iterations": ITERS,
                "ours_ms": ours_ms,
                "flaggems_ms": flaggems_ms,
                "speedup_vs_flaggems": flaggems_ms / ours_ms,
            }
            rows.append(row)
            print(row, flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "flaggems_commit": "30bfcc0735b8ebd48bd7339ef752ec64df69663f",
        "rows": rows,
    }
    output = Path("paper/results/raw/h100_sm90_flaggems_comparison_2026-08-23.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
