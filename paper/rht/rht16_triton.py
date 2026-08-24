"""Batched H16 random Hadamard transform prototype for the NVFP4 Wgrad path."""

import torch
import triton
import triton.language as tl


# Fixed Rademacher sign vector, shared across all rows/layers in this prototype.
DEFAULT_SIGN_MASK = 0xA3F5


@triton.jit
def _rht16_kernel(
    x_ptr,
    y_ptr,
    rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SIGN_MASK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, 16)
    n = tl.arange(0, 16)
    x = tl.load(x_ptr + row[:, None] * 16 + k[None, :], mask=row[:, None] < rows)

    # R = S H / sqrt(16), so x R applies the sign before the Hadamard.
    shared_bits = k[:, None] & n[None, :]
    parity = (
        (shared_bits & 1)
        ^ ((shared_bits >> 1) & 1)
        ^ ((shared_bits >> 2) & 1)
        ^ ((shared_bits >> 3) & 1)
    )
    sign_bit = (SIGN_MASK >> k[:, None]) & 1
    r = tl.where((parity ^ sign_bit) != 0, -0.25, 0.25).to(x.dtype)
    y = tl.dot(x, r, out_dtype=tl.float32).to(x.dtype)
    tl.store(y_ptr + row[:, None] * 16 + n[None, :], y, mask=row[:, None] < rows)


def rht16(x: torch.Tensor, sign_mask: int = DEFAULT_SIGN_MASK) -> torch.Tensor:
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("rht16 requires a CUDA FP16 or BF16 tensor")
    if x.shape[-1] != 16 or not x.is_contiguous():
        raise ValueError("rht16 requires a contiguous tensor with last dimension 16")
    rows = x.numel() // 16
    y = torch.empty_like(x)
    block_m = 16
    _rht16_kernel[(triton.cdiv(rows, block_m),)](
        x,
        y,
        rows=rows,
        BLOCK_M=block_m,
        SIGN_MASK=sign_mask,
        num_warps=4,
    )
    return y


@triton.jit
def _rht16_transpose_kernel(
    x_ptr,
    y_ptr,
    rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SIGN_MASK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, 16)
    n = tl.arange(0, 16)
    x = tl.load(x_ptr + row[:, None] * 16 + k[None, :], mask=row[:, None] < rows)
    shared_bits = k[:, None] & n[None, :]
    parity = (
        (shared_bits & 1)
        ^ ((shared_bits >> 1) & 1)
        ^ ((shared_bits >> 2) & 1)
        ^ ((shared_bits >> 3) & 1)
    )
    # R^T = H S / 4: apply the random sign to each output column.
    sign_bit = (SIGN_MASK >> n[None, :]) & 1
    rt = tl.where((parity ^ sign_bit) != 0, -0.25, 0.25).to(x.dtype)
    y = tl.dot(x, rt, out_dtype=tl.float32).to(x.dtype)
    tl.store(y_ptr + row[:, None] * 16 + n[None, :], y, mask=row[:, None] < rows)


def rht16_transpose(x: torch.Tensor, sign_mask: int = DEFAULT_SIGN_MASK) -> torch.Tensor:
    """Apply the transpose/inverse of the normalized randomized H16 transform."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("rht16_transpose requires a CUDA FP16 or BF16 tensor")
    if x.shape[-1] != 16 or not x.is_contiguous():
        raise ValueError("rht16_transpose requires a contiguous tensor with last dimension 16")
    rows = x.numel() // 16
    y = torch.empty_like(x)
    block_m = 16
    _rht16_transpose_kernel[(triton.cdiv(rows, block_m),)](
        x,
        y,
        rows=rows,
        BLOCK_M=block_m,
        SIGN_MASK=sign_mask,
        num_warps=4,
    )
    return y


class _RHT16Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.input_shape = x.shape
        return rht16(x.contiguous().reshape(-1, 16)).reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = rht16_transpose(grad_output.contiguous().reshape(-1, 16))
        return grad.reshape(ctx.input_shape)


def rht16_autograd(x: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled RHT with the exact transpose operation in backward."""
    if not x.requires_grad:
        raise ValueError("rht16_autograd requires input.requires_grad=True")
    return _RHT16Autograd.apply(x)


@triton.jit
def _rht16_fp8_kernel(
    x_ptr,
    q_ptr,
    inv_scale_ptr,
    rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SIGN_MASK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, 16)
    n = tl.arange(0, 16)
    mask = row[:, None] < rows
    x = tl.load(x_ptr + row[:, None] * 16 + k[None, :], mask=mask)
    shared_bits = k[:, None] & n[None, :]
    parity = (
        (shared_bits & 1)
        ^ ((shared_bits >> 1) & 1)
        ^ ((shared_bits >> 2) & 1)
        ^ ((shared_bits >> 3) & 1)
    )
    sign_bit = (SIGN_MASK >> k[:, None]) & 1
    r = tl.where((parity ^ sign_bit) != 0, -0.25, 0.25).to(x.dtype)
    y = tl.dot(x, r, out_dtype=tl.float32)

    # One E4M3 scale per 16-element row. q * inv_scale reconstructs y.
    amax = tl.max(tl.abs(y), axis=1)
    inv_scale = tl.maximum(amax / 448.0, 1.0e-12)
    tl.store(q_ptr + row[:, None] * 16 + n[None, :], y / inv_scale[:, None], mask=mask)
    tl.store(inv_scale_ptr + row, inv_scale, mask=row < rows)


def rht16_fp8(
    x: torch.Tensor, sign_mask: int = DEFAULT_SIGN_MASK
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("rht16_fp8 requires a CUDA FP16 or BF16 tensor")
    if x.shape[-1] != 16 or not x.is_contiguous():
        raise ValueError("rht16_fp8 requires a contiguous tensor with last dimension 16")
    rows = x.numel() // 16
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    inv_scale = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    block_m = 16
    _rht16_fp8_kernel[(triton.cdiv(rows, block_m),)](
        x,
        q,
        inv_scale,
        rows=rows,
        BLOCK_M=block_m,
        SIGN_MASK=sign_mask,
        num_warps=4,
    )
    return q, inv_scale


def reference_matrix(
    *, device: torch.device, dtype: torch.dtype, sign_mask: int = DEFAULT_SIGN_MASK
) -> torch.Tensor:
    idx = torch.arange(16, device=device, dtype=torch.int64)
    bits = idx[:, None].bitwise_and(idx[None, :])
    parity = torch.zeros_like(bits)
    for shift in range(4):
        parity.bitwise_xor_((bits >> shift) & 1)
    signs = torch.where(((sign_mask >> idx) & 1).bool(), -1.0, 1.0)
    h = torch.where(parity.bool(), -1.0, 1.0)
    return (signs[:, None] * h * 0.25).to(dtype)
