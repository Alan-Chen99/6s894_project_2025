"""Fused H16 RHT and Transformer Engine-compatible FP8 block-128 storage."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rht16_triton import DEFAULT_SIGN_MASK, rht16_transpose


@triton.jit
def _rht16_te_block_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    rows: tl.constexpr,
    padded_rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SIGN_MASK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    block_k = tl.program_id(1)
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)

    shared_bits = i[:, None] & j[None, :]
    parity = (
        (shared_bits & 1)
        ^ ((shared_bits >> 1) & 1)
        ^ ((shared_bits >> 2) & 1)
        ^ ((shared_bits >> 3) & 1)
    )
    sign_bit = (SIGN_MASK >> i[:, None]) & 1
    h = tl.where((parity ^ sign_bit) != 0, -0.25, 0.25)

    base = row[:, None] * tl.num_programs(1) * 128 + block_k * 128
    mask = row[:, None] < rows
    x0 = tl.load(x_ptr + base + 0 * 16 + i[None, :], mask=mask)
    x1 = tl.load(x_ptr + base + 1 * 16 + i[None, :], mask=mask)
    x2 = tl.load(x_ptr + base + 2 * 16 + i[None, :], mask=mask)
    x3 = tl.load(x_ptr + base + 3 * 16 + i[None, :], mask=mask)
    x4 = tl.load(x_ptr + base + 4 * 16 + i[None, :], mask=mask)
    x5 = tl.load(x_ptr + base + 5 * 16 + i[None, :], mask=mask)
    x6 = tl.load(x_ptr + base + 6 * 16 + i[None, :], mask=mask)
    x7 = tl.load(x_ptr + base + 7 * 16 + i[None, :], mask=mask)
    h = h.to(x0.dtype)
    y0 = tl.dot(x0, h, out_dtype=tl.float32)
    y1 = tl.dot(x1, h, out_dtype=tl.float32)
    y2 = tl.dot(x2, h, out_dtype=tl.float32)
    y3 = tl.dot(x3, h, out_dtype=tl.float32)
    y4 = tl.dot(x4, h, out_dtype=tl.float32)
    y5 = tl.dot(x5, h, out_dtype=tl.float32)
    y6 = tl.dot(x6, h, out_dtype=tl.float32)
    y7 = tl.dot(x7, h, out_dtype=tl.float32)

    amax = tl.maximum(tl.max(tl.abs(y0), axis=1), tl.max(tl.abs(y1), axis=1))
    amax = tl.maximum(amax, tl.max(tl.abs(y2), axis=1))
    amax = tl.maximum(amax, tl.max(tl.abs(y3), axis=1))
    amax = tl.maximum(amax, tl.max(tl.abs(y4), axis=1))
    amax = tl.maximum(amax, tl.max(tl.abs(y5), axis=1))
    amax = tl.maximum(amax, tl.max(tl.abs(y6), axis=1))
    amax = tl.maximum(amax, tl.max(tl.abs(y7), axis=1))
    # TE's default block recipe constrains inverse scales to powers of two.
    scale_inv = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1.0e-12))))

    tl.store(q_ptr + base + 0 * 16 + j[None, :], y0 / scale_inv[:, None], mask=mask)
    tl.store(q_ptr + base + 1 * 16 + j[None, :], y1 / scale_inv[:, None], mask=mask)
    tl.store(q_ptr + base + 2 * 16 + j[None, :], y2 / scale_inv[:, None], mask=mask)
    tl.store(q_ptr + base + 3 * 16 + j[None, :], y3 / scale_inv[:, None], mask=mask)
    tl.store(q_ptr + base + 4 * 16 + j[None, :], y4 / scale_inv[:, None], mask=mask)
    tl.store(q_ptr + base + 5 * 16 + j[None, :], y5 / scale_inv[:, None], mask=mask)
    tl.store(q_ptr + base + 6 * 16 + j[None, :], y6 / scale_inv[:, None], mask=mask)
    tl.store(q_ptr + base + 7 * 16 + j[None, :], y7 / scale_inv[:, None], mask=mask)
    tl.store(scale_ptr + block_k * padded_rows + row, scale_inv, mask=row < rows)


def rht16_te_block_buffers(
    x: torch.Tensor, sign_mask: int = DEFAULT_SIGN_MASK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return TE rowwise FP8 bytes and block-128 inverse scales."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("input must be a CUDA FP16 or BF16 tensor")
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] % 128:
        raise ValueError("input must be contiguous [M,K] with K divisible by 128")
    rows, width = x.shape
    if rows % 128:
        raise ValueError("TE block-scaled GEMM requires M divisible by 128")
    padded_rows = triton.cdiv(rows, 4) * 4
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale_inv = torch.empty((width // 128, padded_rows), device=x.device, dtype=torch.float32)
    block_m = 16
    _rht16_te_block_kernel[(triton.cdiv(rows, block_m), width // 128)](
        x,
        q,
        scale_inv,
        rows=rows,
        padded_rows=padded_rows,
        BLOCK_M=block_m,
        SIGN_MASK=sign_mask,
        num_warps=4,
    )
    return q.view(torch.uint8), scale_inv


@triton.jit
def _rht16_te_block_columnwise_kernel(
    x_ptr,
    q_col_ptr,
    scale_col_ptr,
    width: tl.constexpr,
    rows: tl.constexpr,
    padded_width: tl.constexpr,
    SIGN_MASK: tl.constexpr,
):
    row_block = tl.program_id(0)
    group_k = tl.program_id(1)
    row = row_block * 128 + tl.arange(0, 128)
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)
    x = tl.load(x_ptr + row[:, None] * width + group_k * 16 + i[None, :])

    shared_bits = i[:, None] & j[None, :]
    parity = (
        (shared_bits & 1)
        ^ ((shared_bits >> 1) & 1)
        ^ ((shared_bits >> 2) & 1)
        ^ ((shared_bits >> 3) & 1)
    )
    sign_bit = (SIGN_MASK >> i[:, None]) & 1
    h = tl.where((parity ^ sign_bit) != 0, -0.25, 0.25).to(x.dtype)
    y = tl.dot(x, h, out_dtype=tl.float32)
    amax = tl.max(tl.abs(y), axis=0)
    scale_inv = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1.0e-12))))

    col = group_k * 16 + j
    # TE stores columnwise bytes as a materialized [K,M] transpose.
    tl.store(q_col_ptr + row[:, None] + col[None, :] * rows, y / scale_inv[None, :])
    tl.store(scale_col_ptr + row_block * padded_width + col, scale_inv)


def rht16_te_block_columnwise_buffers(
    x: torch.Tensor, sign_mask: int = DEFAULT_SIGN_MASK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return TE columnwise FP8 bytes and block-128 inverse scales."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("input must be a CUDA FP16 or BF16 tensor")
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] % 128:
        raise ValueError("input must be contiguous [M,K] with K divisible by 128")
    rows, width = x.shape
    if rows % 128:
        raise ValueError("TE block-scaled GEMM requires M divisible by 128")
    padded_width = triton.cdiv(width, 4) * 4
    q_col = torch.empty((width, rows), device=x.device, dtype=torch.float8_e4m3fn)
    scale_col = torch.empty((rows // 128, padded_width), device=x.device, dtype=torch.float32)
    _rht16_te_block_columnwise_kernel[(rows // 128, width // 16)](
        x,
        q_col,
        scale_col,
        width=width,
        rows=rows,
        padded_width=padded_width,
        SIGN_MASK=sign_mask,
        num_warps=8,
    )
    return q_col.view(torch.uint8), scale_col


def rht16_te_block_tensor(x: torch.Tensor, quantizer=None, *, columnwise: bool = False):
    """Create a TE Float8BlockwiseQTensor without a requantization kernel."""
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockwiseQTensor,
    )

    if quantizer is None:
        quantizer = te.Float8BlockQuantizer(
            te.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=columnwise,
            block_scaling_dim=1,
        )
    data, scale_inv = rht16_te_block_buffers(x)
    columnwise_data = None
    columnwise_scale_inv = None
    if columnwise:
        columnwise_data, columnwise_scale_inv = rht16_te_block_columnwise_buffers(x)
    return Float8BlockwiseQTensor(
        shape=x.shape,
        dtype=x.dtype,
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise_data=data,
        rowwise_scale_inv=scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        quantizer=quantizer,
        is_2D_scaled=False,
        device=x.device,
    )


class _RHT16TEBlockAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.input_shape = x.shape
        return rht16_te_block_tensor(x, columnwise=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = rht16_transpose(grad_output.contiguous().reshape(-1, 16))
        return grad.reshape(ctx.input_shape)


def rht16_te_block_autograd(x: torch.Tensor):
    """Quantized RHT tensor whose Dgrad is mapped back through R^T."""
    if not x.requires_grad:
        raise ValueError("autograd adapter requires input.requires_grad=True")
    return _RHT16TEBlockAutograd.apply(x)
