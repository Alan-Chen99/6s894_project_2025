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
    SWIGLU: tl.constexpr,
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

    width = tl.num_programs(1) * 128
    input_width = 2 * width if SWIGLU else width
    base = row[:, None] * input_width + block_k * 128
    mask = row[:, None] < rows
    x0 = tl.load(x_ptr + base + 0 * 16 + i[None, :], mask=mask)
    x1 = tl.load(x_ptr + base + 1 * 16 + i[None, :], mask=mask)
    x2 = tl.load(x_ptr + base + 2 * 16 + i[None, :], mask=mask)
    x3 = tl.load(x_ptr + base + 3 * 16 + i[None, :], mask=mask)
    x4 = tl.load(x_ptr + base + 4 * 16 + i[None, :], mask=mask)
    x5 = tl.load(x_ptr + base + 5 * 16 + i[None, :], mask=mask)
    x6 = tl.load(x_ptr + base + 6 * 16 + i[None, :], mask=mask)
    x7 = tl.load(x_ptr + base + 7 * 16 + i[None, :], mask=mask)
    if SWIGLU:
        u0 = tl.load(x_ptr + base + width + 0 * 16 + i[None, :], mask=mask)
        u1 = tl.load(x_ptr + base + width + 1 * 16 + i[None, :], mask=mask)
        u2 = tl.load(x_ptr + base + width + 2 * 16 + i[None, :], mask=mask)
        u3 = tl.load(x_ptr + base + width + 3 * 16 + i[None, :], mask=mask)
        u4 = tl.load(x_ptr + base + width + 4 * 16 + i[None, :], mask=mask)
        u5 = tl.load(x_ptr + base + width + 5 * 16 + i[None, :], mask=mask)
        u6 = tl.load(x_ptr + base + width + 6 * 16 + i[None, :], mask=mask)
        u7 = tl.load(x_ptr + base + width + 7 * 16 + i[None, :], mask=mask)
        x0 = (x0.to(tl.float32) * tl.sigmoid(x0.to(tl.float32)) * u0).to(x0.dtype)
        x1 = (x1.to(tl.float32) * tl.sigmoid(x1.to(tl.float32)) * u1).to(x1.dtype)
        x2 = (x2.to(tl.float32) * tl.sigmoid(x2.to(tl.float32)) * u2).to(x2.dtype)
        x3 = (x3.to(tl.float32) * tl.sigmoid(x3.to(tl.float32)) * u3).to(x3.dtype)
        x4 = (x4.to(tl.float32) * tl.sigmoid(x4.to(tl.float32)) * u4).to(x4.dtype)
        x5 = (x5.to(tl.float32) * tl.sigmoid(x5.to(tl.float32)) * u5).to(x5.dtype)
        x6 = (x6.to(tl.float32) * tl.sigmoid(x6.to(tl.float32)) * u6).to(x6.dtype)
        x7 = (x7.to(tl.float32) * tl.sigmoid(x7.to(tl.float32)) * u7).to(x7.dtype)
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
    # A 64-row tile gives substantially better SM90 occupancy than 16 rows for
    # this eight-dot writer while keeping register pressure below the cliff.
    block_m = 64
    _rht16_te_block_kernel[(triton.cdiv(rows, block_m), width // 128)](
        x,
        q,
        scale_inv,
        rows=rows,
        padded_rows=padded_rows,
        BLOCK_M=block_m,
        SIGN_MASK=sign_mask,
        SWIGLU=False,
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
    SWIGLU: tl.constexpr,
):
    row_block = tl.program_id(0)
    group_k = tl.program_id(1)
    row = row_block * 128 + tl.arange(0, 128)
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)
    input_width = 2 * width if SWIGLU else width
    base = row[:, None] * input_width + group_k * 16 + i[None, :]
    x = tl.load(x_ptr + base)
    if SWIGLU:
        up = tl.load(x_ptr + base + width)
        x = (x.to(tl.float32) * tl.sigmoid(x.to(tl.float32)) * up).to(x.dtype)

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
        SWIGLU=False,
        num_warps=4,
    )
    return q_col.view(torch.uint8), scale_col


@triton.jit
def _rht16_te_block_both_kernel(
    x_ptr,
    q_ptr,
    row_scale_ptr,
    q_col_ptr,
    col_scale_ptr,
    rows: tl.constexpr,
    padded_rows: tl.constexpr,
    padded_width: tl.constexpr,
    SIGN_MASK: tl.constexpr,
    SWIGLU: tl.constexpr,
    INPUT_FP32: tl.constexpr,
    SCALE_2D: tl.constexpr,
):
    """Write both TE block-scaled views from one 128x128 RHT tile."""
    row_block = tl.program_id(0)
    block_k = tl.program_id(1)
    row = row_block * 128 + tl.arange(0, 128)
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

    width = tl.num_programs(1) * 128
    input_width = 2 * width if SWIGLU else width
    base = row[:, None] * input_width + block_k * 128
    x0 = tl.load(x_ptr + base + 0 * 16 + i[None, :])
    x1 = tl.load(x_ptr + base + 1 * 16 + i[None, :])
    x2 = tl.load(x_ptr + base + 2 * 16 + i[None, :])
    x3 = tl.load(x_ptr + base + 3 * 16 + i[None, :])
    x4 = tl.load(x_ptr + base + 4 * 16 + i[None, :])
    x5 = tl.load(x_ptr + base + 5 * 16 + i[None, :])
    x6 = tl.load(x_ptr + base + 6 * 16 + i[None, :])
    x7 = tl.load(x_ptr + base + 7 * 16 + i[None, :])
    if INPUT_FP32:
        x0 = x0.to(tl.bfloat16)
        x1 = x1.to(tl.bfloat16)
        x2 = x2.to(tl.bfloat16)
        x3 = x3.to(tl.bfloat16)
        x4 = x4.to(tl.bfloat16)
        x5 = x5.to(tl.bfloat16)
        x6 = x6.to(tl.bfloat16)
        x7 = x7.to(tl.bfloat16)
    if SWIGLU:
        u0 = tl.load(x_ptr + base + width + 0 * 16 + i[None, :])
        u1 = tl.load(x_ptr + base + width + 1 * 16 + i[None, :])
        u2 = tl.load(x_ptr + base + width + 2 * 16 + i[None, :])
        u3 = tl.load(x_ptr + base + width + 3 * 16 + i[None, :])
        u4 = tl.load(x_ptr + base + width + 4 * 16 + i[None, :])
        u5 = tl.load(x_ptr + base + width + 5 * 16 + i[None, :])
        u6 = tl.load(x_ptr + base + width + 6 * 16 + i[None, :])
        u7 = tl.load(x_ptr + base + width + 7 * 16 + i[None, :])
        x0 = (x0.to(tl.float32) * tl.sigmoid(x0.to(tl.float32)) * u0).to(x0.dtype)
        x1 = (x1.to(tl.float32) * tl.sigmoid(x1.to(tl.float32)) * u1).to(x1.dtype)
        x2 = (x2.to(tl.float32) * tl.sigmoid(x2.to(tl.float32)) * u2).to(x2.dtype)
        x3 = (x3.to(tl.float32) * tl.sigmoid(x3.to(tl.float32)) * u3).to(x3.dtype)
        x4 = (x4.to(tl.float32) * tl.sigmoid(x4.to(tl.float32)) * u4).to(x4.dtype)
        x5 = (x5.to(tl.float32) * tl.sigmoid(x5.to(tl.float32)) * u5).to(x5.dtype)
        x6 = (x6.to(tl.float32) * tl.sigmoid(x6.to(tl.float32)) * u6).to(x6.dtype)
        x7 = (x7.to(tl.float32) * tl.sigmoid(x7.to(tl.float32)) * u7).to(x7.dtype)
    h = h.to(x0.dtype)
    y0 = tl.dot(x0, h, out_dtype=tl.float32)
    y1 = tl.dot(x1, h, out_dtype=tl.float32)
    y2 = tl.dot(x2, h, out_dtype=tl.float32)
    y3 = tl.dot(x3, h, out_dtype=tl.float32)
    y4 = tl.dot(x4, h, out_dtype=tl.float32)
    y5 = tl.dot(x5, h, out_dtype=tl.float32)
    y6 = tl.dot(x6, h, out_dtype=tl.float32)
    y7 = tl.dot(x7, h, out_dtype=tl.float32)

    # Match the normal FP32-master -> BF16 working-weight boundary before
    # quantization. Activation inputs are already BF16/FP16 and retain their
    # FP32 dot accumulator through the fused quantizer.
    if INPUT_FP32:
        y0 = y0.to(tl.bfloat16)
        y1 = y1.to(tl.bfloat16)
        y2 = y2.to(tl.bfloat16)
        y3 = y3.to(tl.bfloat16)
        y4 = y4.to(tl.bfloat16)
        y5 = y5.to(tl.bfloat16)
        y6 = y6.to(tl.bfloat16)
        y7 = y7.to(tl.bfloat16)

    row_amax = tl.maximum(tl.max(tl.abs(y0), axis=1), tl.max(tl.abs(y1), axis=1))
    row_amax = tl.maximum(row_amax, tl.max(tl.abs(y2), axis=1))
    row_amax = tl.maximum(row_amax, tl.max(tl.abs(y3), axis=1))
    row_amax = tl.maximum(row_amax, tl.max(tl.abs(y4), axis=1))
    row_amax = tl.maximum(row_amax, tl.max(tl.abs(y5), axis=1))
    row_amax = tl.maximum(row_amax, tl.max(tl.abs(y6), axis=1))
    row_amax = tl.maximum(row_amax, tl.max(tl.abs(y7), axis=1))
    row_scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(row_amax / 448.0, 1.0e-12))))

    col_scale0 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(y0), axis=0) / 448.0, 1.0e-12))))
    col_scale1 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(y1), axis=0) / 448.0, 1.0e-12))))
    col_scale2 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(y2), axis=0) / 448.0, 1.0e-12))))
    col_scale3 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(y3), axis=0) / 448.0, 1.0e-12))))
    col_scale4 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(y4), axis=0) / 448.0, 1.0e-12))))
    col_scale5 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(y5), axis=0) / 448.0, 1.0e-12))))
    col_scale6 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(y6), axis=0) / 448.0, 1.0e-12))))
    col_scale7 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(y7), axis=0) / 448.0, 1.0e-12))))
    if SCALE_2D:
        tile_scale = tl.exp2(
            tl.ceil(tl.log2(tl.maximum(tl.max(row_amax, axis=0) / 448.0, 1.0e-12)))
        )
        row_scale = tile_scale
        col_scale0 = tile_scale
        col_scale1 = tile_scale
        col_scale2 = tile_scale
        col_scale3 = tile_scale
        col_scale4 = tile_scale
        col_scale5 = tile_scale
        col_scale6 = tile_scale
        col_scale7 = tile_scale

    tl.store(q_ptr + base + 0 * 16 + j[None, :], y0 / row_scale[:, None])
    tl.store(q_ptr + base + 1 * 16 + j[None, :], y1 / row_scale[:, None])
    tl.store(q_ptr + base + 2 * 16 + j[None, :], y2 / row_scale[:, None])
    tl.store(q_ptr + base + 3 * 16 + j[None, :], y3 / row_scale[:, None])
    tl.store(q_ptr + base + 4 * 16 + j[None, :], y4 / row_scale[:, None])
    tl.store(q_ptr + base + 5 * 16 + j[None, :], y5 / row_scale[:, None])
    tl.store(q_ptr + base + 6 * 16 + j[None, :], y6 / row_scale[:, None])
    tl.store(q_ptr + base + 7 * 16 + j[None, :], y7 / row_scale[:, None])
    if SCALE_2D:
        tl.store(row_scale_ptr + row_block * padded_width + block_k, tile_scale)
    else:
        tl.store(row_scale_ptr + block_k * padded_rows + row, row_scale)

    col_base = block_k * 128 + j
    tl.store(q_col_ptr + row[:, None] + (col_base + 0 * 16)[None, :] * rows, y0 / col_scale0[None, :])
    tl.store(q_col_ptr + row[:, None] + (col_base + 1 * 16)[None, :] * rows, y1 / col_scale1[None, :])
    tl.store(q_col_ptr + row[:, None] + (col_base + 2 * 16)[None, :] * rows, y2 / col_scale2[None, :])
    tl.store(q_col_ptr + row[:, None] + (col_base + 3 * 16)[None, :] * rows, y3 / col_scale3[None, :])
    tl.store(q_col_ptr + row[:, None] + (col_base + 4 * 16)[None, :] * rows, y4 / col_scale4[None, :])
    tl.store(q_col_ptr + row[:, None] + (col_base + 5 * 16)[None, :] * rows, y5 / col_scale5[None, :])
    tl.store(q_col_ptr + row[:, None] + (col_base + 6 * 16)[None, :] * rows, y6 / col_scale6[None, :])
    tl.store(q_col_ptr + row[:, None] + (col_base + 7 * 16)[None, :] * rows, y7 / col_scale7[None, :])
    if SCALE_2D:
        tl.store(col_scale_ptr + block_k * padded_rows + row_block, tile_scale)
    else:
        scale_base = row_block * padded_width + block_k * 128 + j
        tl.store(col_scale_ptr + scale_base + 0 * 16, col_scale0)
        tl.store(col_scale_ptr + scale_base + 1 * 16, col_scale1)
        tl.store(col_scale_ptr + scale_base + 2 * 16, col_scale2)
        tl.store(col_scale_ptr + scale_base + 3 * 16, col_scale3)
        tl.store(col_scale_ptr + scale_base + 4 * 16, col_scale4)
        tl.store(col_scale_ptr + scale_base + 5 * 16, col_scale5)
        tl.store(col_scale_ptr + scale_base + 6 * 16, col_scale6)
        tl.store(col_scale_ptr + scale_base + 7 * 16, col_scale7)


def rht16_te_block_both_buffers(
    x: torch.Tensor, sign_mask: int = DEFAULT_SIGN_MASK
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rowwise and columnwise TE buffers from one RHT evaluation."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("input must be a CUDA FP16, BF16, or FP32 tensor")
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] % 128:
        raise ValueError("input must be contiguous [M,K] with K divisible by 128")
    rows, width = x.shape
    if rows % 128:
        raise ValueError("TE block-scaled GEMM requires M divisible by 128")
    padded_rows = triton.cdiv(rows, 4) * 4
    padded_width = triton.cdiv(width, 4) * 4
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((width // 128, padded_rows), device=x.device, dtype=torch.float32)
    q_col = torch.empty((width, rows), device=x.device, dtype=torch.float8_e4m3fn)
    col_scale = torch.empty((rows // 128, padded_width), device=x.device, dtype=torch.float32)
    _rht16_te_block_both_kernel[(rows // 128, width // 128)](
        x,
        q,
        row_scale,
        q_col,
        col_scale,
        rows=rows,
        padded_rows=padded_rows,
        padded_width=padded_width,
        SIGN_MASK=sign_mask,
        SWIGLU=False,
        INPUT_FP32=x.dtype == torch.float32,
        SCALE_2D=False,
        num_warps=4,
    )
    return q.view(torch.uint8), row_scale, q_col.view(torch.uint8), col_scale


def rht16_te_block_both_into(
    x: torch.Tensor,
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    columnwise_data: torch.Tensor,
    columnwise_scale_inv: torch.Tensor,
    sign_mask: int = DEFAULT_SIGN_MASK,
) -> None:
    """Write fused RHT block-FP8 data directly into an existing TE tensor."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("input must be a CUDA FP16, BF16, or FP32 tensor")
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] % 128:
        raise ValueError("input must be contiguous [M,K] with K divisible by 128")
    rows, width = x.shape
    if rows % 128:
        raise ValueError("TE block-scaled GEMM requires M divisible by 128")
    padded_rows = triton.cdiv(rows, 4) * 4
    padded_width = triton.cdiv(width, 4) * 4
    expected = (
        (rows, width),
        (width // 128, padded_rows),
        (width, rows),
        (rows // 128, padded_width),
    )
    actual = (
        tuple(rowwise_data.shape),
        tuple(rowwise_scale_inv.shape),
        tuple(columnwise_data.shape),
        tuple(columnwise_scale_inv.shape),
    )
    if actual != expected:
        raise ValueError(f"TE buffer shapes {actual} do not match expected {expected}")
    if any(not tensor.is_contiguous() for tensor in (
        rowwise_data,
        rowwise_scale_inv,
        columnwise_data,
        columnwise_scale_inv,
    )):
        raise ValueError("TE output buffers must be contiguous")
    row_fp8 = (
        rowwise_data.view(torch.float8_e4m3fn)
        if rowwise_data.dtype == torch.uint8
        else rowwise_data
    )
    col_fp8 = (
        columnwise_data.view(torch.float8_e4m3fn)
        if columnwise_data.dtype == torch.uint8
        else columnwise_data
    )
    if row_fp8.dtype != torch.float8_e4m3fn or col_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("TE data buffers must contain E4M3 bytes")
    _rht16_te_block_both_kernel[(rows // 128, width // 128)](
        x,
        row_fp8,
        rowwise_scale_inv,
        col_fp8,
        columnwise_scale_inv,
        rows=rows,
        padded_rows=padded_rows,
        padded_width=padded_width,
        SIGN_MASK=sign_mask,
        SWIGLU=False,
        INPUT_FP32=x.dtype == torch.float32,
        SCALE_2D=False,
        num_warps=4,
    )


def rht16_te_block_2d_into(
    x: torch.Tensor,
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    columnwise_data: torch.Tensor,
    columnwise_scale_inv: torch.Tensor,
    sign_mask: int = DEFAULT_SIGN_MASK,
) -> None:
    """Write RHT weights into TE's native 128x128 2D block-scaled layout."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("input must be a CUDA FP16, BF16, or FP32 tensor")
    if x.ndim != 2 or not x.is_contiguous() or any(value % 128 for value in x.shape):
        raise ValueError("input must be contiguous [M,K], both divisible by 128")
    rows, width = x.shape
    row_blocks, width_blocks = rows // 128, width // 128
    padded_row_blocks = triton.cdiv(row_blocks, 4) * 4
    padded_width_blocks = triton.cdiv(width_blocks, 4) * 4
    expected = (
        (rows, width),
        (row_blocks, padded_width_blocks),
        (width, rows),
        (width_blocks, padded_row_blocks),
    )
    actual = (
        tuple(rowwise_data.shape),
        tuple(rowwise_scale_inv.shape),
        tuple(columnwise_data.shape),
        tuple(columnwise_scale_inv.shape),
    )
    if actual != expected:
        raise ValueError(f"2D TE buffer shapes {actual} do not match expected {expected}")
    row_fp8 = (
        rowwise_data.view(torch.float8_e4m3fn)
        if rowwise_data.dtype == torch.uint8
        else rowwise_data
    )
    col_fp8 = (
        columnwise_data.view(torch.float8_e4m3fn)
        if columnwise_data.dtype == torch.uint8
        else columnwise_data
    )
    _rht16_te_block_both_kernel[(row_blocks, width_blocks)](
        x,
        row_fp8,
        rowwise_scale_inv,
        col_fp8,
        columnwise_scale_inv,
        rows=rows,
        padded_rows=padded_row_blocks,
        padded_width=padded_width_blocks,
        SIGN_MASK=sign_mask,
        SWIGLU=False,
        INPUT_FP32=x.dtype == torch.float32,
        SCALE_2D=True,
        num_warps=4,
    )


def swiglu_rht16_te_block_both_buffers(
    x: torch.Tensor, sign_mask: int = DEFAULT_SIGN_MASK
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse SwiGLU, RHT, and both TE block-scaled output views."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("input must be a CUDA FP16 or BF16 tensor")
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] % 256:
        raise ValueError("input must be contiguous [M,2K] with K divisible by 128")
    rows, double_width = x.shape
    width = double_width // 2
    if rows % 128:
        raise ValueError("TE block-scaled GEMM requires M divisible by 128")
    padded_rows = triton.cdiv(rows, 4) * 4
    padded_width = triton.cdiv(width, 4) * 4
    q = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((width // 128, padded_rows), device=x.device, dtype=torch.float32)
    q_col = torch.empty((width, rows), device=x.device, dtype=torch.float8_e4m3fn)
    col_scale = torch.empty((rows // 128, padded_width), device=x.device, dtype=torch.float32)
    block_m = 64
    _rht16_te_block_kernel[(triton.cdiv(rows, block_m), width // 128)](
        x,
        q,
        row_scale,
        rows=rows,
        padded_rows=padded_rows,
        BLOCK_M=block_m,
        SIGN_MASK=sign_mask,
        SWIGLU=True,
        num_warps=4,
    )
    _rht16_te_block_columnwise_kernel[(rows // 128, width // 16)](
        x,
        q_col,
        col_scale,
        width=width,
        rows=rows,
        padded_width=padded_width,
        SIGN_MASK=sign_mask,
        SWIGLU=True,
        num_warps=4,
    )
    return q.view(torch.uint8), row_scale, q_col.view(torch.uint8), col_scale


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
    columnwise_data = None
    columnwise_scale_inv = None
    if columnwise:
        data, scale_inv, columnwise_data, columnwise_scale_inv = rht16_te_block_both_buffers(x)
    else:
        data, scale_inv = rht16_te_block_buffers(x)
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


def swiglu_rht16_te_block_tensor(x: torch.Tensor, quantizer=None):
    """Create a TE tensor directly from a fused SwiGLU and RHT input."""
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockwiseQTensor,
    )

    if quantizer is None:
        quantizer = te.Float8BlockQuantizer(
            te.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=1,
        )
    data, scale_inv, columnwise_data, columnwise_scale_inv = (
        swiglu_rht16_te_block_both_buffers(x)
    )
    shape = (x.shape[0], x.shape[1] // 2)
    return Float8BlockwiseQTensor(
        shape=shape,
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


@triton.jit
def _rht16_transpose_dswiglu_kernel(
    grad_ptr,
    input_ptr,
    grad_input_ptr,
    rows: tl.constexpr,
    width: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SIGN_MASK: tl.constexpr,
):
    """Fuse inverse RHT with the SwiGLU derivative."""
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group_k = tl.program_id(1)
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)
    grad = tl.load(grad_ptr + row[:, None] * width + group_k * 16 + i[None, :])

    shared_bits = i[:, None] & j[None, :]
    parity = (
        (shared_bits & 1)
        ^ ((shared_bits >> 1) & 1)
        ^ ((shared_bits >> 2) & 1)
        ^ ((shared_bits >> 3) & 1)
    )
    sign_bit = (SIGN_MASK >> j[None, :]) & 1
    rt = tl.where((parity ^ sign_bit) != 0, -0.25, 0.25).to(grad.dtype)
    grad_hidden = tl.dot(grad, rt, out_dtype=tl.float32)

    input_width = 2 * width
    base = row[:, None] * input_width + group_k * 16 + j[None, :]
    gate = tl.load(input_ptr + base).to(tl.float32)
    up = tl.load(input_ptr + base + width).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    silu = gate * sigmoid
    grad_gate = grad_hidden * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    grad_up = grad_hidden * silu
    tl.store(grad_input_ptr + base, grad_gate)
    tl.store(grad_input_ptr + base + width, grad_up)


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


class _SwiGLURHT16TEBlockAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        # TE's dedicated SwiGLU kernel is faster on SM90 than folding sigmoid
        # into the register-heavy bidirectional writer. Keep that forward path
        # and fuse the inverse RHT with dSwiGLU in backward below.
        import transformer_engine_torch as tex

        hidden = tex.swiglu(x, None)
        return rht16_te_block_tensor(hidden, columnwise=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        rows, double_width = x.shape
        width = double_width // 2
        grad_input = torch.empty_like(x)
        block_m = 64
        _rht16_transpose_dswiglu_kernel[
            (triton.cdiv(rows, block_m), width // 16)
        ](
            grad_output.contiguous(),
            x,
            grad_input,
            rows=rows,
            width=width,
            BLOCK_M=block_m,
            SIGN_MASK=DEFAULT_SIGN_MASK,
            num_warps=4,
        )
        return grad_input


def swiglu_rht16_te_block_autograd(x: torch.Tensor):
    """TE SwiGLU + RHT/FP8 forward with fused inverse-RHT SwiGLU backward."""
    if not x.requires_grad:
        raise ValueError("autograd adapter requires input.requires_grad=True")
    return _SwiGLURHT16TEBlockAutograd.apply(x)
