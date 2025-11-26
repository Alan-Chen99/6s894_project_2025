#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

torch::Tensor test_mma_m16_n8_k16(at::Tensor& A, at::Tensor& B);

auto test_bit_convert(at::Tensor& A) -> std::tuple<at::Tensor, at::Tensor>;
