"""Original-basis optimizer masters for dynamically paired RHT weights."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl

from rht16_triton import DEFAULT_SIGN_MASK, rht16_into, rht16_transpose_into


@dataclass
class DynamicWeightBridge:
    """Connect an FP32 optimizer master to a BF16/FP16 TE working weight.

    Adam moments stay attached to ``master`` in the original basis. Before a
    forward pass, ``materialize`` writes either W or W R into ``working``.
    After backward, ``map_grad_to_master`` writes either G or G R^T into the
    master gradient. The rotated case is therefore mathematically correct for
    elementwise optimizers such as Adam; no invariance assumption is needed.
    """

    master: torch.nn.Parameter
    working: torch.nn.Parameter
    rotated: bool

    @classmethod
    def from_working_weight(
        cls,
        working: torch.nn.Parameter,
        *,
        rotated: bool,
        initial: torch.Tensor | None = None,
    ) -> "DynamicWeightBridge":
        if working.ndim != 2 or working.shape[1] % 16:
            raise ValueError("working weight must be [out,in] with in divisible by 16")
        source = working.detach() if initial is None else initial.detach()
        if source.shape != working.shape:
            raise ValueError("initial and working weight shapes must match")
        # Always own storage: callers commonly initialize several matched paths
        # from the same already-contiguous FP32 tensor.
        master = torch.nn.Parameter(source.float().contiguous().clone())
        return cls(master=master, working=working, rotated=rotated)

    @torch.no_grad()
    def materialize(self) -> None:
        """Refresh the TE working weight from the current optimizer master."""
        if self.rotated:
            rht16_into(
                self.master.reshape(-1, 16),
                self.working.reshape(-1, 16),
            )
        else:
            self.working.copy_(self.master)

    @torch.no_grad()
    def map_grad_to_master(self) -> None:
        """Map the TE working gradient into the optimizer's original basis."""
        if self.working.grad is None:
            raise RuntimeError("working weight gradient is missing")
        if self.master.grad is None:
            self.master.grad = torch.empty_like(self.master)
        if self.rotated:
            rht16_transpose_into(
                self.working.grad.contiguous().reshape(-1, 16),
                self.master.grad.reshape(-1, 16),
            )
        else:
            self.master.grad.copy_(self.working.grad)

    def clear_working_grad(self) -> None:
        self.working.grad = None


@dataclass
class DynamicQuantizedWeightBridge:
    """Original-basis FP32 master paired with a directly written TE FP8 weight."""

    master: torch.nn.Parameter
    working: torch.nn.Parameter

    @classmethod
    def attach(
        cls,
        layer: torch.nn.Module,
        initial: torch.Tensor,
    ) -> "DynamicQuantizedWeightBridge":
        if initial.ndim != 2 or any(value % 128 for value in initial.shape):
            raise ValueError("quantized weight dimensions must be divisible by 128")
        # Do not alias an FP32 contiguous initializer shared by control paths.
        master = torch.nn.Parameter(initial.detach().float().contiguous().clone())
        working = layer.weight
        if not hasattr(working, "_is_2D_scaled") or not working._is_2D_scaled:
            raise ValueError(
                "layer must be created under te.quantized_model_init with block scaling"
            )
        return cls(master=master, working=working)

    @torch.no_grad()
    def materialize(self) -> None:
        from rht16_te_block import rht16_te_block_2d_into

        rht16_te_block_2d_into(
            self.master,
            self.working._rowwise_data,
            self.working._rowwise_scale_inv,
            self.working._columnwise_data,
            self.working._columnwise_scale_inv,
        )

    @torch.no_grad()
    def map_grad_to_master(self) -> None:
        if self.working.grad is None:
            raise RuntimeError("quantized working weight gradient is missing")
        if self.master.grad is None:
            self.master.grad = torch.empty_like(self.master)
        rht16_transpose_into(
            self.working.grad.contiguous().reshape(-1, 16),
            self.master.grad.reshape(-1, 16),
        )

    def clear_working_grad(self) -> None:
        self.working.grad = None


WeightBridge = DynamicWeightBridge | DynamicQuantizedWeightBridge


def materialize_all(bridges: list[WeightBridge]) -> None:
    for bridge in bridges:
        bridge.materialize()


def map_all_grads(bridges: list[WeightBridge]) -> None:
    for bridge in bridges:
        bridge.map_grad_to_master()


def clear_all_working_grads(bridges: list[WeightBridge]) -> None:
    for bridge in bridges:
        bridge.clear_working_grad()


@triton.jit
def _rht16_transpose_adamw_kernel(
    grad_ptr,
    master_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SIGN_MASK: tl.constexpr,
    BETA1,
    BETA2,
    STEP_SIZE,
    INV_SQRT_BIAS_CORRECTION2,
    LR,
    WEIGHT_DECAY,
    EPS,
):
    """Fuse inverse H16, FP32 Adam moments, and decoupled weight decay."""
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)
    mask = row[:, None] < rows
    grad = tl.load(grad_ptr + row[:, None] * 16 + i[None, :], mask=mask)

    shared_bits = i[:, None] & j[None, :]
    parity = (
        (shared_bits & 1)
        ^ ((shared_bits >> 1) & 1)
        ^ ((shared_bits >> 2) & 1)
        ^ ((shared_bits >> 3) & 1)
    )
    sign_bit = (SIGN_MASK >> j[None, :]) & 1
    rt = tl.where((parity ^ sign_bit) != 0, -0.25, 0.25).to(grad.dtype)
    grad_master = tl.dot(grad, rt, out_dtype=tl.float32)

    offsets = row[:, None] * 16 + j[None, :]
    master = tl.load(master_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
    exp_avg = BETA1 * exp_avg + (1.0 - BETA1) * grad_master
    exp_avg_sq = BETA2 * exp_avg_sq + (1.0 - BETA2) * grad_master * grad_master
    master = master * (1.0 - LR * WEIGHT_DECAY)
    denom = tl.sqrt(exp_avg_sq) * INV_SQRT_BIAS_CORRECTION2 + EPS
    master = master - STEP_SIZE * exp_avg / denom
    tl.store(master_ptr + offsets, master, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


class FusedRHTAdamW:
    """AdamW whose update consumes rotated BF16 Wgrad without a map buffer."""

    def __init__(
        self,
        bridges: list[DynamicQuantizedWeightBridge],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if not bridges:
            raise ValueError("FusedRHTAdamW requires at least one weight bridge")
        self.bridges = bridges
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.state = [
            {
                "exp_avg": torch.zeros_like(bridge.master),
                "exp_avg_sq": torch.zeros_like(bridge.master),
            }
            for bridge in bridges
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        del set_to_none
        clear_all_working_grads(self.bridges)

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count
        step_size = self.lr / bias_correction1
        inv_sqrt_bias_correction2 = 1.0 / math.sqrt(bias_correction2)
        for bridge, state in zip(self.bridges, self.state):
            grad = bridge.working.grad
            if grad is None:
                raise RuntimeError("quantized working weight gradient is missing")
            if not grad.is_contiguous():
                raise RuntimeError("fused AdamW requires contiguous working gradients")
            rows = grad.numel() // 16
            block_m = 16
            _rht16_transpose_adamw_kernel[(triton.cdiv(rows, block_m),)](
                grad.reshape(-1, 16),
                bridge.master.reshape(-1, 16),
                state["exp_avg"].reshape(-1, 16),
                state["exp_avg_sq"].reshape(-1, 16),
                rows=rows,
                BLOCK_M=block_m,
                SIGN_MASK=DEFAULT_SIGN_MASK,
                BETA1=self.beta1,
                BETA2=self.beta2,
                STEP_SIZE=step_size,
                INV_SQRT_BIAS_CORRECTION2=inv_sqrt_bias_correction2,
                LR=self.lr,
                WEIGHT_DECAY=self.weight_decay,
                EPS=self.eps,
                num_warps=4,
            )
