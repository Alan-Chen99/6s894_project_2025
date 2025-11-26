#include "frag.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

template <DType dtype>
__global__ void test_mma_m16_n8_k16_ker(const u16* a, const u16* b, u16* c)
{
    int lane = threadIdx.x;
    assert_(0 <= lane && lane < 32);

    auto a_frag = Frag<dtype, 8, NoMeta, array<NoMeta, 8>{}, PermA>::template load<
        array<int, 8>{1, 2, 4, 8, 16, 32, 64, 128}, // row major
        MemCheckNone,
        1
    >(a, lane);

    auto b_frag = Frag<dtype, 7, NoMeta, array<NoMeta, 7>{}, PermB>::template load<
        array<int, 7>{8, 16, 32, 64, 1, 2, 4}, // row major
        MemCheckNone,
        1
    >(b, lane);

    Frag<dtype, 7, NoMeta, array<NoMeta, 7>{}, PermC> c_frag =
        mma_m16_n8_k16<dtype>(a_frag, b_frag);

    c_frag.template store<
        array<int, 7>{8, 16, 32, 64, 1, 2, 4}, // row major
        MemCheckNone,
        1
    >(c, lane);
}

torch::Tensor test_mma_m16_n8_k16(at::Tensor& A, at::Tensor& B)
{
    A = A.contiguous();
    B = B.contiguous();

    TORCH_CHECK_EQ(A.sizes(), at::IntArrayRef({16, 16}));
    TORCH_CHECK_EQ(B.sizes(), at::IntArrayRef({16, 8}));

    TORCH_CHECK(A.device().is_cuda());
    TORCH_CHECK(B.device() == A.device());

    TORCH_CHECK_EQ(A.scalar_type(), torch::ScalarType::Half);
    TORCH_CHECK_EQ(B.scalar_type(), torch::ScalarType::Half);

    at::cuda::CUDAGuard device_guard(A.get_device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto C = torch::empty_like(B);

    test_mma_m16_n8_k16_ker<DType::Half><<<1, 32, 0, stream>>>(
        static_cast<const u16*>(A.data_ptr()),
        static_cast<const u16*>(B.data_ptr()),
        static_cast<u16*>(C.data_ptr())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}
