#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

template <torch::ScalarType dtype>
void run_fht(void* a, void* out, uint32_t numel, uint32_t had_size, cudaStream_t stream);

torch::Tensor test_mma_m16_n8_k16(at::Tensor& A, at::Tensor& B);
