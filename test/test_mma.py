import torch

from csrc import test_mma_m16_n8_k16

A = torch.rand(16, 16, device="cuda", dtype=torch.float16)
B = torch.rand(16, 8, device="cuda", dtype=torch.float16)

C = test_mma_m16_n8_k16(A, B)
assert torch.allclose(A @ B, C, atol=1e-2)
