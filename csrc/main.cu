// comment out to enable assert
#define NDEBUG

#include "hada_handler.cuh"
#include "main.cuh"
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cstdio>

// had dimension is 2**N
template <DType dtype, int N>
__global__ __launch_bounds__(32, 4) auto hadamard_transform_ker(const u16* a, u16* out)
    -> void
{
    int lane = threadIdx.x;
    assert_((0 <= lane) && (lane < 32));

    using H = RowHandler<dtype, N>;

    constexpr int R = H::rows_per_block;
    constexpr int S = H::log_elem_count;

    constexpr int elem_per_block = (1 << S) * R;

    // start pointers of what this block handles
    const u16* in_base = a + blockIdx.x * elem_per_block;
    u16* out_base = out + blockIdx.x * elem_per_block;

    extern __shared__ u16 sm[];

#pragma unroll
    for (int i = 0; i < R; i++) {

        const u16* a_i = in_base + i * (1 << S);
        u16* out_i = out_base + i * (1 << S);

        H::handle_row(a_i, out_i, lane, sm);
    }
}

__device__ void example_device_call(const u16* in, u16* out)
{
    using Transform = HadamardTransform<DType::Half, 8>;

    // Transform::T is the # of element of one call
    // Dummy2<Transform::T>::x x;

    // static, or pass a portion of dynamic shared memory
    __shared__ u8 sm[std::max(Transform::SM_BYTES, 1)];

    Transform::hadamard_transform_device(in, out, sm);
}

template <DType dtype, int N>
auto run_fht_helper(void* a_mat_ptr, void* out_ptr, uint32_t numel, cudaStream_t stream)
    -> void
{
    using H = RowHandler<dtype, N>;

    // # of elements to process in one block
    constexpr int elem_per_block = (1 << H::log_elem_count) * H::rows_per_block;

    TORCH_CHECK(numel % elem_per_block == 0);

    auto ker = hadamard_transform_ker<dtype, N>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ker,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        H::SM_BYTES
    ));

    ker<<<numel / elem_per_block, 32, H::SM_BYTES, stream>>>(
        static_cast<const u16*>(a_mat_ptr),
        static_cast<u16*>(out_ptr)
    );
}

template <DType dtype>
auto run_fht(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
) -> void
{
    // TORCH_CHECK(had_size == 256, "This implementation only supports Hadamard size
    // 256");

    // TORCH_CHECK(numel % had_size == 0);
    // int num_rows = numel / had_size;

    // constexpr int R = 4;
    // TORCH_CHECK(num_rows % R == 0);

    bool found = false;

    static_for<16>([&]<int I>() {
        if (had_size == (1 << I)) {
            if constexpr (size_supported(I)) {
                found = true;
                run_fht_helper<dtype, I>(a_mat_ptr, out_ptr, numel, stream);
            }
        }
    });

    TORCH_CHECK(found, "unsupported size");

    // TORCH_CHECK((numel % 256) == 0, "numel must be divisible by 256");

    // const uint32_t num_rows = numel / 256;
    // dim3 grid(num_rows), block(32);
    // constexpr size_t shmem = 128 * sizeof(u32); // 512B

    // hadamard_transform_256_kernel<dtype><<<grid, block, shmem, stream>>>(
    //     static_cast<const u16*>(a_mat_ptr),
    //     static_cast<u16*>(out_ptr),
    //     static_cast<int>(num_rows)
    // );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Explicit instantiations (fp16, bf16)
template auto run_fht<DType::Half>(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
) -> void;

template auto run_fht<DType::BFloat16>(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
) -> void;
