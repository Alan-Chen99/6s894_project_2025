import dataclasses
import functools
import json
import math
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Callable
from typing import Protocol

import cappa
import numpy as np
import scipy.linalg
import torch

# from rich import print

import csrc
import fast_hadamard_transform_cuda
import hada_core


class BenchmarkTarget(Protocol):
    """
    A protocol for a benchmarkable operation.

    This defines the interface that all benchmarks must follow.
    It requires an optimized `call` and a reference `call_ref` for correctness checks.
    Both methods must perform their operations in-place.
    """

    def call(self, t: torch.Tensor) -> None:
        """The optimized, in-place function to benchmark."""
        ...

    def call_ref(self, t: torch.Tensor) -> None:
        """The reference in-place implementation for correctness checks."""
        ...


def get_scale(size: float) -> float:
    return math.sqrt(1 / size)


@functools.cache
def _hadamard_cache(size: int) -> torch.Tensor:
    return torch.tensor(
        scipy.linalg.hadamard(size), device="cuda", dtype=torch.float32
    ) * get_scale(size)


def hadamard_ref(t: torch.Tensor) -> None:
    m = int(t.shape[-1])

    ans = t @ _hadamard_cache(m).to(t.dtype)
    t.copy_(ans)


class FastHada(BenchmarkTarget):
    def call(self, t: torch.Tensor) -> None:
        t.copy_(fast_hadamard_transform_cuda.fast_hadamard_transform(t, get_scale(t.shape[-1])))

    def call_ref(self, t: torch.Tensor) -> None:
        hadamard_ref(t)


class HadaCore(BenchmarkTarget):
    def call(self, t: torch.Tensor) -> None:
        hada_core.hadamard_transform(t, inplace=True)

    def call_ref(self, t: torch.Tensor) -> None:
        hadamard_ref(t)


class OwnHada(BenchmarkTarget):
    def call(self, t: torch.Tensor) -> None:
        csrc.hadamard_transform(t, inplace=True)

    def call_ref(self, t: torch.Tensor) -> None:
        HadaCore().call(t)
        # hadamard_ref(t)


class AddOne(BenchmarkTarget):
    def call(self, t: torch.Tensor) -> None:
        t.add_(1)

    def call_ref(self, t: torch.Tensor) -> None:
        t.add_(1)


CASES: dict[str, BenchmarkTarget] = {
    "AddOne": AddOne(),
    # "FastHada": FastHada(),
    "HadaCore": HadaCore(),
    "OwnHada": OwnHada(),
}

##########


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _bench_many(fn: Callable[[], None], iters: int, warmup: int = 3) -> dict[str, float]:
    # Assume CUDA. Time many runs on the same buffer with one event window.
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    total_ms: float = float(start.elapsed_time(end))
    per_iter_ms: float = total_ms / max(iters, 1)
    return {"total_ms": total_ms, "per_iter_ms": per_iter_ms}


# hadamard sizes
# test_sizes_m = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
test_sizes_m = [256]

test_elem_counts = [1 << i for i in range(9, 26, 1)]  # 32MB # 64MB # 2**28 = 256M


@dataclass
class TestConfig:
    """Configuration for the test suite."""

    check: bool = False
    runs_per_size: int = 100
    json_output_file: str | None = f"benchmark_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"

    benchmark_cases: list[str] = field(default_factory=lambda: list(CASES.keys()))

    test_sizes: list[int] = field(default_factory=lambda: list(test_sizes_m))
    elem_counts: list[int] = field(default_factory=lambda: list(test_elem_counts))
    # dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)
    # only one for faster build
    dtypes: tuple[torch.dtype, ...] = (torch.float16,)


def _run_checks(
    a: torch.Tensor,
    m: int,
    n: int,
    cfg: TestConfig,
    target: BenchmarkTarget,
    atol_map: dict[torch.dtype, float],
) -> None:
    for _ in range(cfg.runs_per_size):
        for dtype in cfg.dtypes:
            a_result = a.clone().to(dtype)
            a_truth = a.clone().to(dtype)

            target.call(a_result)
            target.call_ref(a_truth)

            atol = atol_map.get(dtype, 1e-2)
            success = torch.allclose(a_truth, a_result, atol=atol, rtol=0)
            if success:
                continue

            torch.set_printoptions(threshold=100)
            print(f"Failed test: {m}x{n}")
            print("Input:")
            print(a)
            print("Expected:")
            print(a_truth)
            print("Got:")
            print(a_result)

            diff = torch.abs(a_truth - a_result)
            print("diff:", diff)
            # print("diff:", diff.tolist())

            max_diff = torch.max(diff)
            print(f"Max diff: {max_diff}")

            flat_idx = torch.argmax(diff)
            coords = torch.unravel_index(flat_idx, diff.shape)
            print(f"Max diff index: {[x.item() for x in coords]}")

            # diff_input = torch.abs(a.to(a_result.dtype) - a_result)
            # max_diff_input = torch.max(diff_input)
            # print(f"Max diff input: {max_diff_input}")
            print()
            raise SystemExit(1)


def _run_perf(
    a: torch.Tensor,
    m: int,
    n: int,
    cfg: TestConfig,
    target: BenchmarkTarget,
    device_name: str,
    case_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dtype in cfg.dtypes:
        a_run = a.to(dtype).contiguous()

        stats = _bench_many(lambda: target.call(a_run), cfg.runs_per_size)

        print(
            f"{case_name} m={m} n={n} {_dtype_name(dtype)} | "
            f"{stats['per_iter_ms']:.3f} ms/iter over {cfg.runs_per_size} iters"
        )

        out.append(
            {
                "device": device_name,
                "case": case_name,
                "m": m,
                "n": n,
                "dtype": _dtype_name(dtype),
                "runs": cfg.runs_per_size,
                "per_iter_ms": stats["per_iter_ms"],
                "total_ms": stats["total_ms"],
            }
        )
    return out


def main(cfg: TestConfig) -> None:
    # selected cases
    case_names = cfg.benchmark_cases or list(CASES.keys())

    # echo config like the reference script
    print("test_sizes_m: ", cfg.test_sizes)
    print("test_elem_counts: ", cfg.elem_counts)
    print("cases: ", case_names)

    # progress counters (count each size pair once)
    test_count = len(cfg.test_sizes) * len(cfg.elem_counts)
    tests_done = 0

    # torch setup
    torch.manual_seed(0)
    device_name: str = torch.cuda.get_device_name(0)

    # tolerances match the reference script
    atol_map: dict[torch.dtype, float] = {
        torch.float16: 1e-2,
        torch.bfloat16: 5e-2,
    }

    results: list[dict[str, Any]] = []

    for m in cfg.test_sizes:
        for elem_c in cfg.elem_counts:
            n = elem_c // m

            if elem_c >= m:
                print(f"Testing size {m}x{n}")

                # base input in float32 on CUDA
                a = torch.randn((n, m), device="cuda", dtype=torch.float32)

                for case_name in case_names:
                    target = CASES[case_name]

                    _run_checks(a, m, n, cfg, target, atol_map)

                    if not cfg.check:
                        results.extend(_run_perf(a, m, n, cfg, target, device_name, case_name))

            # one increment per size pair (fixed counter)
            tests_done += 1
            if tests_done % 100 == 0 or tests_done == test_count:
                print(f"{tests_done}/{test_count} size tests done")

    if not cfg.check and cfg.json_output_file:
        payload: dict[str, Any] = {
            "device": device_name,
            "cases": case_names,
            "pytorch": torch.__version__,
            "results": results,
        }
        with open(cfg.json_output_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {cfg.json_output_file}")


if __name__ == "__main__":
    main(cappa.parse(TestConfig))
