#!/usr/bin/env python3
"""Short matched convergence run for BF16, block-FP8, and dynamic RHT FP8."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.ops.basic import SwiGLU

sys.path.insert(0, str(Path("paper/rht").resolve()))
from dynamic_weight import (  # noqa: E402
    DynamicQuantizedWeightBridge,
    DynamicWeightBridge,
    FusedRHTAdamW,
    clear_all_working_grads,
    map_all_grads,
    materialize_all,
)
from rht16_te_block import (  # noqa: E402
    rht16_te_block_autograd,
    swiglu_rht16_te_block_autograd,
)


class TrainingPath:
    def __init__(
        self,
        name: str,
        initial_w1: torch.Tensor,
        initial_w2: torch.Tensor,
        *,
        fp8: bool,
        rotated: bool,
        fused_rht_adamw: bool,
        learning_rate: float,
    ) -> None:
        hidden = initial_w1.shape[1]
        intermediate = initial_w2.shape[1]
        self.name = name
        self.fp8 = fp8
        self.rotated = rotated
        self.fused_rht_adamw = fused_rht_adamw
        self.block_recipe = recipe.Float8BlockScaling()
        init_context = (
            te.quantized_model_init(enabled=True, recipe=self.block_recipe)
            if rotated
            else nullcontext()
        )
        with init_context:
            self.fc1 = te.Linear(
                hidden,
                2 * intermediate,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
            )
            self.down = te.Linear(
                intermediate,
                hidden,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
            )
        self.swiglu = SwiGLU()
        if rotated:
            self.bridges = [
                DynamicQuantizedWeightBridge.attach(self.fc1, initial_w1),
                DynamicQuantizedWeightBridge.attach(self.down, initial_w2),
            ]
        else:
            self.bridges = [
                DynamicWeightBridge.from_working_weight(
                    self.fc1.weight, rotated=False, initial=initial_w1
                ),
                DynamicWeightBridge.from_working_weight(
                    self.down.weight, rotated=False, initial=initial_w2
                ),
            ]
        if fused_rht_adamw:
            if not rotated:
                raise ValueError("fused RHT AdamW requires rotated weights")
            self.optimizer = FusedRHTAdamW(
                self.bridges,
                lr=learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                [bridge.master for bridge in self.bridges],
                lr=learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        materialize_all(self.bridges)
        context = (
            te.autocast(enabled=True, recipe=self.block_recipe)
            if self.fp8
            else nullcontext()
        )
        with context:
            if self.rotated:
                qx = rht16_te_block_autograd(x)
                fc1_output = self.fc1(qx)
                hidden = swiglu_rht16_te_block_autograd(fc1_output)
            else:
                fc1_output = self.fc1(x)
                hidden = self.swiglu(fc1_output)
            return self.down(hidden)

    def step(self, x: torch.Tensor, target: torch.Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        clear_all_working_grads(self.bridges)
        output = self.forward(x)
        loss = F.mse_loss(output.float(), target.float())
        loss.backward()
        if not self.fused_rht_adamw:
            map_all_grads(self.bridges)
        self.optimizer.step()
        return float(loss.detach())


@torch.no_grad()
def teacher_output(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    z = x.float() @ w1.T
    gate, up = z.chunk(2, dim=-1)
    return ((F.silu(gate) * up) @ w2.T).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--dataset-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--optimizer-comparison-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m, k, n = args.tokens, args.hidden, args.intermediate
    if any(value % 128 for value in (m, k, n)):
        raise ValueError("tokens, hidden, and intermediate must be divisible by 128")

    torch.manual_seed(2027)
    teacher_w1 = torch.randn(2 * n, k, device="cuda") / math.sqrt(k)
    teacher_w2 = torch.randn(k, n, device="cuda") / math.sqrt(n)
    initial_w1 = teacher_w1 + 0.08 * torch.randn_like(teacher_w1) / math.sqrt(k)
    initial_w2 = teacher_w2 + 0.08 * torch.randn_like(teacher_w2) / math.sqrt(n)
    inputs = [
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        for _ in range(args.dataset_batches)
    ]
    targets = [teacher_output(x, teacher_w1, teacher_w2) for x in inputs]

    optimizer_paths = [
        TrainingPath(
            "dynamic_rht_block_fp8",
            initial_w1,
            initial_w2,
            fp8=True,
            rotated=True,
            fused_rht_adamw=False,
            learning_rate=args.learning_rate,
        ),
        TrainingPath(
            "dynamic_rht_block_fp8_fused_adamw",
            initial_w1,
            initial_w2,
            fp8=True,
            rotated=True,
            fused_rht_adamw=True,
            learning_rate=args.learning_rate,
        ),
    ]
    if args.optimizer_comparison_only:
        # Construct only the paired RHT paths. TE initialization owns global
        # state, so even unused controls can perturb an optimizer A/B test.
        paths = optimizer_paths
    else:
        control_paths = [
            TrainingPath(
                "bf16",
                initial_w1,
                initial_w2,
                fp8=False,
                rotated=False,
                fused_rht_adamw=False,
                learning_rate=args.learning_rate,
            ),
            TrainingPath(
                "block_fp8",
                initial_w1,
                initial_w2,
                fp8=True,
                rotated=False,
                fused_rht_adamw=False,
                learning_rate=args.learning_rate,
            ),
        ]
        paths = [*control_paths, *optimizer_paths]

    records = {path.name: {"loss": [], "step_ms": []} for path in paths}
    # Warm all TE/Triton paths without changing optimizer state.
    for path in paths:
        x = inputs[0].detach().requires_grad_(True)
        output = path.forward(x)
        output.float().sum().backward()
        clear_all_working_grads(path.bridges)
    torch.cuda.synchronize()

    for step in range(args.steps):
        batch = step % args.dataset_batches
        for path in paths:
            x = inputs[batch].detach().requires_grad_(True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = path.step(x, targets[batch])
            end.record()
            end.synchronize()
            records[path.name]["loss"].append(loss)
            records[path.name]["step_ms"].append(start.elapsed_time(end))
        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"step {step + 1:3d}: "
                + " ".join(
                    f"{path.name}={records[path.name]['loss'][-1]:.6g}"
                    for path in paths
                )
            )

    summary = {}
    timing_start = min(10, max(1, args.steps // 4))
    for path in paths:
        path_record = records[path.name]
        median_ms = statistics.median(path_record["step_ms"][timing_start:])
        summary[path.name] = {
            "initial_loss": path_record["loss"][0],
            "final_loss": path_record["loss"][-1],
            "loss_reduction": path_record["loss"][0] / path_record["loss"][-1],
            "median_step_ms": median_ms,
            "tokens_per_second": m / (median_ms * 1.0e-3),
        }
    if "bf16" in summary:
        bf16_final = summary["bf16"]["final_loss"]
        for name in (
            "block_fp8",
            "dynamic_rht_block_fp8",
            "dynamic_rht_block_fp8_fused_adamw",
        ):
            summary[name]["final_loss_ratio_vs_bf16"] = (
                summary[name]["final_loss"] / bf16_final
            )
    mapped_final = summary["dynamic_rht_block_fp8"]["final_loss"]
    summary["dynamic_rht_block_fp8_fused_adamw"][
        "final_loss_ratio_vs_mapped_adamw"
    ] = (
        summary["dynamic_rht_block_fp8_fused_adamw"]["final_loss"]
        / mapped_final
    )

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformer_engine_version": transformer_engine.__version__,
        "shape": {"tokens": m, "hidden": k, "intermediate": n},
        "steps": args.steps,
        "dataset_batches": args.dataset_batches,
        "optimizer_comparison_only": args.optimizer_comparison_only,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "master_dtype": "float32",
            "state_basis": "original/unrotated",
        },
        "task": "matched teacher-student SwiGLU MLP regression",
        "summary": summary,
        "curves": records,
        "scope_note": (
            "Short controlled convergence test, not language-model pretraining. "
            "All paths use FP32 original-basis masters; dynamic RHT fuses WR with "
            "2D block-FP8 weight storage. The fused optimizer consumes rotated "
            "Wgrad directly; its control maps Wgrad through R^T before AdamW."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
