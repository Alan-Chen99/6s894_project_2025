"""Correctness and saturated-throughput benchmark for the H16 RHT prototype."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("paper/rht").resolve()))
from rht16_triton import reference_matrix, rht16  # noqa: E402


DTYPES = (torch.float16, torch.bfloat16)
ROWS = 1 << 22
WARMUP = 20
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
    rows = []
    for dtype in DTYPES:
        check = torch.randn(1024, 16, device="cuda", dtype=dtype)
        matrix = reference_matrix(device=check.device, dtype=dtype)
        actual = rht16(check)
        expected = check @ matrix
        max_error = float((actual - expected).abs().max())

        x = torch.randn(ROWS, 16, device="cuda", dtype=dtype)
        # Preallocate/reference expression still dispatches a matmul and output.
        triton_ms = time_many(lambda: rht16(x))
        torch_ms = time_many(lambda: torch.matmul(x, matrix))
        row = {
            "dtype": str(dtype).removeprefix("torch."),
            "rows": ROWS,
            "elements": ROWS * 16,
            "correctness_max_error": max_error,
            "triton_rht16_ms": triton_ms,
            "torch_matmul_ms": torch_ms,
            "speedup_vs_torch_matmul": torch_ms / triton_ms,
        }
        rows.append(row)
        print(row, flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "operation": "x @ (S @ H16 / 4), fixed sign mask 0xA3F5",
        "rows": rows,
    }
    output = Path("paper/results/raw/h100_sm90_rht16_2026-08-23.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
