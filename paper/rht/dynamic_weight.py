"""Original-basis optimizer masters for dynamically paired RHT weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rht16_triton import rht16_into, rht16_transpose_into


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
        master = torch.nn.Parameter(source.float().contiguous())
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
        master = torch.nn.Parameter(initial.detach().float().contiguous())
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
