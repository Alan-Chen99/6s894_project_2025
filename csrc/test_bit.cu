#include "utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

__global__ void test_bit_ker(const float* x, u16* bits_f16, u16* bits_bf16)
{
    *bits_f16 = f32_to_dtype<DType::Half>(*x);
    *bits_bf16 = f32_to_dtype<DType::BFloat16>(*x);
}

auto test_bit_convert(at::Tensor& A) -> std::tuple<at::Tensor, at::Tensor>
{
    TORCH_CHECK(A.numel() == 1, "test_bit_convert expects a scalar tensor");

    // Ensure we have a CUDA device to run on
    int dev_index = at::cuda::current_device();
    c10::Device dev(c10::kCUDA, dev_index);
    c10::cuda::CUDAGuard guard(dev);

    // Move/cast input to CUDA float32 scalar
    at::Tensor A_cuda = A.to(at::kCUDA, at::kFloat).contiguous();

    // Allocate scalar outputs on the same CUDA device
    auto bf16_out = at::empty({}, A_cuda.options().dtype(at::kBFloat16));
    auto f16_out = at::empty({}, A_cuda.options().dtype(at::kHalf));

    // Raw device pointers
    const float* x_ptr = A_cuda.data_ptr<float>();
    auto* f16_bits = reinterpret_cast<u16*>(f16_out.data_ptr<c10::Half>());
    auto* bf16_bits = reinterpret_cast<u16*>(bf16_out.data_ptr<c10::BFloat16>());

    // Launch
    auto stream = at::cuda::getCurrentCUDAStream();
    test_bit_ker<<<1, 1, 0, stream.stream()>>>(x_ptr, f16_bits, bf16_bits);
    C10_CUDA_CHECK(cudaGetLastError());

    // Return (bfloat16, float16) scalars
    return {bf16_out, f16_out};
}
