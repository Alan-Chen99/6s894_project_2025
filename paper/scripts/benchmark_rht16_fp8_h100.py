"""Benchmark fused tiled RHT + block scale + E4M3 cast on Hopper."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("paper/rht").resolve()))
from rht16_triton import rht16, rht16_fp8  # noqa: E402


DTYPES = (torch.float16, torch.bfloat16)
ROWS = 1 << 22
WARMUP = 20
ITERS = 400


def separate(x):
    y = rht16(x)
    inv_scale = (y.float().abs().amax(dim=-1) / 448.0).clamp_min(1.0e-12)
    q = (y.float() / inv_scale[:, None]).to(torch.float8_e4m3fn)
    return q, inv_scale


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
    output_rows = []
    for dtype in DTYPES:
        check = torch.randn(4096, 16, device="cuda", dtype=dtype)
        truth = rht16(check).float()
        q, inv_scale = rht16_fp8(check)
        reconstructed = q.float() * inv_scale[:, None]
        mse = float(torch.mean((truth - reconstructed) ** 2))
        max_error = float((truth - reconstructed).abs().max())

        x = torch.randn(ROWS, 16, device="cuda", dtype=dtype)
        fused_ms = time_many(lambda: rht16_fp8(x))
        separate_ms = time_many(lambda: separate(x))
        row = {
            "dtype": str(dtype).removeprefix("torch."),
            "rows": ROWS,
            "elements": ROWS * 16,
            "fused_ms": fused_ms,
            "separate_ms": separate_ms,
            "speedup_fused_vs_separate": separate_ms / fused_ms,
            "reconstruction_mse": mse,
            "reconstruction_max_error": max_error,
        }
        output_rows.append(row)
        print(row, flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "operation": "fused H16 RHT + per-16 amax/scale + E4M3 output",
        "rows": output_rows,
    }
    output = Path("paper/results/raw/h100_sm90_rht16_fp8_2026-08-23.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
