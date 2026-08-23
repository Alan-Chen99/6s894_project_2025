"""Matched saturated-throughput comparison with arthurfeeney/fwht."""

import json
import math
import sys
from pathlib import Path

import torch

import csrc


BASELINE_ROOT = Path("paper/baselines/src/arthurfeeney-fwht").resolve()
sys.path.insert(0, str(BASELINE_ROOT))
from fwht import fast_hadamard_transform as arthur_fwht  # noqa: E402


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
            scale = 1.0 / math.sqrt(size)

            # Preflight compilation and compare identical normalized semantics.
            check = torch.randn(32, size, device="cuda", dtype=dtype)
            ours_check = csrc.hadamard_transform(check.clone(), inplace=True)
            try:
                arthur_check = arthur_fwht(check.clone(), scale=scale, inplace=True)
            except Exception as exc:
                error_lines = str(exc).splitlines()
                error = next(
                    (line for line in error_lines if "out_dtype=bfloat16" in line),
                    error_lines[0],
                )
                row = {
                    "dtype": str(dtype).removeprefix("torch."),
                    "size": size,
                    "supported": False,
                    "error_type": type(exc).__name__,
                    "error": error,
                }
                rows.append(row)
                print(row, flush=True)
                continue

            max_error = float((ours_check - arthur_check).abs().max())
            source = torch.randn(batch, size, device="cuda", dtype=dtype)
            ours_input = source.clone()
            arthur_input = source.clone()

            # Both calls mutate their inputs and apply identical normalization.
            ours_ms = time_many(
                lambda: csrc.hadamard_transform(ours_input, inplace=True)
            )
            arthur_ms = time_many(
                lambda: arthur_fwht(arthur_input, scale=scale, inplace=True)
            )
            row = {
                "dtype": str(dtype).removeprefix("torch."),
                "size": size,
                "batch": batch,
                "total_elements": TOTAL_ELEMENTS,
                "iterations": ITERS,
                "supported": True,
                "correctness_max_error": max_error,
                "ours_ms": ours_ms,
                "arthur_ms": arthur_ms,
                "speedup_vs_arthur": arthur_ms / ours_ms,
            }
            rows.append(row)
            print(row, flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "arthur_fwht_commit": "9fcb9fe8dfac000bbebed001809b02b49a55dcdc",
        "local_patch": "BF16 tl.dot uses FP32 output and casts at stage boundaries",
        "rows": rows,
    }
    output = Path("paper/results/raw/h100_sm90_arthur_comparison_2026-08-23.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
