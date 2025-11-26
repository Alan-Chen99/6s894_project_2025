// comment out to enable assert
#define NDEBUG

#include "defs.cuh"
#include "frag.cuh"
#include "hada.cuh"
#include "utils.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

/////////////////////////////////////////////////

// load M * 256 element from global in to shared out
// every 256 consecutive element will be hadamard transformed
template <
    DType dtype,
    int N,
    int P, // # of cp.async pipeline at once
    int S
    // array<int, N> sm_strides // layout in shared mem
>
__device__ auto load_rot_8(const u16* in, u16* sm, int lane) -> void
{
    static_for<N + P>([&]<int I>() {
        if constexpr (I < N) {
            cp_async16(
                sm + I * S + lane * 8,  // shared: stride S
                in + I * 256 + lane * 8 // gobal: stride 256
            );
            async_commit_group();
        }

        if constexpr (P <= I) {
            constexpr int idx = I - P;
            constexpr int ongoing = std::min(I + 1, N);

            async_wait_pending<ongoing - idx - 1>();
            __syncwarp();

            u16* data = sm + idx * S;

            auto in_reg = Frag<
                dtype,
                8,
                AxSpec,
                repeat_to_array<8>(AxSpec::Rot),
                Perm<8>{1, 2, 3, 4, 5, 0 /*u16->u32 packing axis*/, 6, 7}
            >::load(data, lane);
            auto out_reg = hada_rot_8(in_reg, lane);
            out_reg.store(data, lane);
        }
    });
}

/////////////////////////////////////////////////

template <DType dtype, int N> struct RowHandler;

template <DType dtype> struct RowHandler<dtype, 8> {
    static constexpr int rows_per_block = 4;

    __device__ auto handle_row(const u16* in, u16* out) -> void
    {
        int lane = threadIdx.x;
        assert_((0 <= lane) && (lane < 32));

        constexpr auto perm = Perm<8>{
            // axis 1-4 is a coalesced load
            1,
            2,
            3,
            4,
            5,
            // local, assign one axis to pack to u32
            0,
            6,
            7
        };

        // if we controlled order with this instead, loads would not be colaced
        constexpr auto strides = default_strides<8>();

        auto in_reg = Frag<dtype, 8, AxSpec, repeat_to_array<8>(AxSpec::Rot), perm>::
            template load<strides, Coalesced>(in, lane);

        auto out_reg = hada_rot_8(in_reg, lane);

        out_reg.template store<strides, Coalesced>(out, lane);
    }
};

template <DType dtype> struct RowHandler<dtype, 12> {
    static constexpr int rows_per_block = 1;

    __device__ auto handle_row(const u16* in, u16* out) -> void
    {
        int lane = threadIdx.x;
        assert_((0 <= lane) && (lane < 32));

        constexpr int S = 256 + 8;

        __shared__ u16 sm[S * 16];

        load_rot_8<dtype, 16, 7, S>(in, sm, lane);

        __syncwarp();

#pragma unroll
        for (int i = 0; i < 16; i++) {
            u16* base = sm + 16 * i;

            constexpr array<int, 8> strides = {1, 2, 4, 8, S, S * 2, S * 4, S * 8};

            // axis 4, 5, 6, 7 needs rotation
            // -> need to be in position 0, 1, 5, 6
            auto in_reg = Frag<
                dtype,
                8,
                AxSpec,
                array_concat(
                    repeat_to_array<4>(AxSpec::Id),
                    repeat_to_array<4>(AxSpec::Rot)
                ),
                Perm<8>{
                    6 /*0*/,
                    5 /*1*/,
                    3,
                    1,
                    2,
                    4 /*5*/,
                    7 /*6*/,
                    0 /*u16->u32 packing axis*/
                }
            >::template load<strides, SmBankCount<1>>(base, lane);

            Frag<
                dtype,
                8,
                AxSpec,
                repeat_to_array<8>(AxSpec::Id),
                Perm<8>{1, 2, 4, 6, 5, 3, 7, 0}
            >
                out_reg = hada_rot_4(in_reg, lane);

            out_reg.template store<strides, SmBankCount<1>>(base, lane);
        }

        for (int i = 0; i < 16; i++) {
            const u16* ptr_sm = sm + S * i;
            u16* ptr_out = out + 256 * i;

            cp_16(ptr_out + lane * 8, ptr_sm + lane * 8);
        }
    }
};

// had dimension is 2**N
template <DType dtype, int N>
__global__ auto hadamard_transform_ker(const u16* a, u16* out, int num_rows) -> void
{
    RowHandler<dtype, N> handler{};

    constexpr int R = handler.rows_per_block;

#pragma unroll
    for (int j = 0; j < R; j++) {
        int i = blockIdx.x * R + j;

        const u16* a_i = a + i * (1 << N);
        u16* out_i = out + i * (1 << N);

        handler.handle_row(a_i, out_i);
    }

    // for (int i = blockIdx.x; i < num_rows; i += gridDim.x) {
    //     const u16* a_i = a + i * (1 << N);
    //     u16* out_i = out + i * (1 << N);

    //     handler.handle_row(a_i, out_i);
    // }
}

template <torch::ScalarType dtype, int N>
auto run_fht_helper(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t num_rows,
    cudaStream_t stream
) -> void
{
    RowHandler<dtype, N> handler{};

    TORCH_CHECK(num_rows % handler.rows_per_block == 0);

    hadamard_transform_ker<dtype, N>
        <<<num_rows / handler.rows_per_block, 32, 0, stream>>>(
            static_cast<const u16*>(a_mat_ptr),
            static_cast<u16*>(out_ptr),
            num_rows
        );
}

template <torch::ScalarType dtype>
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

    TORCH_CHECK(numel % had_size == 0);
    int num_rows = numel / had_size;

    // constexpr int R = 4;
    // TORCH_CHECK(num_rows % R == 0);

    if (had_size == 256) {
        run_fht_helper<dtype, 8>(a_mat_ptr, out_ptr, num_rows, stream);
    } else if (had_size == 4096) {
        run_fht_helper<dtype, 12>(a_mat_ptr, out_ptr, num_rows, stream);
    }

    // TORCH_CHECK((numel % 256) == 0, "numel must be divisible by 256");

    // const uint32_t num_rows = numel / 256;
    // dim3 grid(num_rows), block(32);
    // constexpr size_t shmem = 128 * sizeof(u32); // 512B

    // hadamard_transform_256_kernel<dtype><<<grid, block, shmem, stream>>>(
    //     static_cast<const u16*>(a_mat_ptr),
    //     static_cast<u16*>(out_ptr),
    //     static_cast<int>(num_rows)
    // );
    // C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Explicit instantiations (fp16, bf16)
template auto run_fht<torch::ScalarType::Half>(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
) -> void;
// template auto run_fht<torch::ScalarType::BFloat16>(
//     void* a_mat_ptr,
//     void* out_ptr,
//     uint32_t numel,
//     uint32_t had_size,
//     cudaStream_t stream
// ) -> void;
