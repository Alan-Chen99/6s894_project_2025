// comment out to enable assert
#define NDEBUG

#include "frag.cuh"
#include "hada.cuh"
#include "main.cuh"
#include "utils.cuh"
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cstdio>

/////////////////////////////////////////////////

// load M * 256 element from global in to shared out
// every 256 consecutive element will be hadamard transformed
template <
    DType dtype,
    int N,
    int P,      // # of cp.async pipeline at once
    int S1,     // first loading into shared
    int S2 = S1 // stride writing back into shared
>
__device__ auto load_rot_8(const u16* in, u16* sm, int lane) -> void
{
    static_assert(S2 <= S1);

    static_for<N + P>([&]<int I>() {
        if constexpr (I < N) {
            cp_async16(
                sm + I * S1 + lane * 8, // shared: stride S
                in + I * 256 + lane * 8 // gobal: stride 256
            );
            async_commit_group();
        }

        if constexpr (P <= I) {
            constexpr int idx = I - P;
            constexpr int ongoing = std::min(I + 1, N);

            async_wait_pending<ongoing - idx - 1>();
            __syncwarp();

            u16* data_in = sm + idx * S1;
            u16* data_out = sm + idx * S2;

            auto in_reg = Frag<
                dtype,
                8,
                AxSpec,
                repeat_to_array<8>(AxSpec::Rot),
                Perm<8>{1, 2, 3, 4, 5, 0 /*u16->u32 packing axis*/, 6, 7}
            >::load(data_in, lane);
            auto out_reg = hada_rot_8(in_reg, lane);
            out_reg.store(data_out, lane);
        }
    });
}

// rotate 0..9 except 3, 4
template <
    DType dtype,
    int N,
    int P // # of cp.async pipeline at once
>
__device__ auto load_rot_8_11(const u16* in, u16* sm, int lane) -> void
{
    static_for<N + P>([&]<int I>() {
        if constexpr (I < N) {
            // copy 2048 contiguous elements
            for (int i = 0; i < 8; i++) {
                int offset = I * 2048 + 256 * i + lane * 8;
                cp_async16(sm + offset, in + offset);
            }
            async_commit_group();
        }

        if constexpr (P <= I) {
            constexpr int idx = I - P;
            constexpr int ongoing = std::min(I + 1, N);

            async_wait_pending<ongoing - idx - 1>();
            __syncwarp();

            u16* data = sm + idx * 2048;

            constexpr array<AxSpec, 11> spec = []() consteval {
                array<AxSpec, 11> spec = repeat_to_array<11>(AxSpec::Id);
                // 6, 7, 8, 9, 10: must rotate
                for (int i = 6; i < 11; i++) {
                    spec[i] = AxSpec::Rot;
                }
                for (int ax : {0, 1, 2}) {
                    spec[ax] = AxSpec::Rot;
                }

                return spec;
            }();

            constexpr auto perm = Perm<11>{
                4,
                5,
                6,
                7,
                8,
                // local, assign one axis to pack to u32
                0,
                9,
                10,
                1,
                2,
                3,
            };

            auto in_reg = Frag<
                dtype,
                11,
                AxSpec,
                spec,
                perm
            >::
                template load<
                    default_strides<11>(), //
                    SmBankCount<2>,
                    8
                >(data, lane);

            auto out_reg = hada_rot_all(in_reg, lane);
            // typename Dummy<decltype(out_reg)>::x x;

            static_assert(rot_finished(out_reg.AxMeta_));

            out_reg.template store<
                default_strides<11>(), //
                SmBankCount<2>,
                8
            >(data, lane);
        }
    });
}

/////////////////////////////////////////////////

// log_elem_count =
// each handle_row call process 1<<log_elem_count elements
// contiguous elements; (longer than 1<<N is batch)
//
// rows_per_block = # of repetition of loop on kernel;

template <DType dtype, int N> struct RowHandler;

template <DType dtype> struct RowHandler<dtype, 8> {
    static constexpr int log_elem_count = 8;
    static constexpr int rows_per_block = 4;

    static constexpr int SM = 0;

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

template <DType dtype, int N>
    requires(8 < N && N <= 12)
struct RowHandler<dtype, N> {
    static constexpr int log_elem_count = 12;
    static constexpr int rows_per_block = 1;

    static constexpr int SM = 0;

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

            // N=12 => axis 4, 5, 6, 7 needs rotation
            // N=(8, 12] => prefix of that
            constexpr array<AxSpec, 8> spec = []() consteval {
                array<AxSpec, 8> spec = repeat_to_array<8>(AxSpec::Id);
                for (int i = 4; i < N - 4; i++) {
                    spec[i] = AxSpec::Rot;
                }
                return spec;
            }();

            // axis 4, 5, 6, 7 needs rotation
            // -> need to be in position 0, 1, 5, 6
            auto in_reg = Frag<
                dtype,
                8,
                AxSpec,
                spec,
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

template <DType dtype, int N>
// requires(12 < N && N <= 15)
    requires(N == 15)
struct RowHandler<dtype, N> {
    static constexpr int log_elem_count = 15;
    static constexpr int rows_per_block = 1;

    static constexpr int SM = (1 << 15) * 2;

    __device__ auto handle_row(const u16* in, u16* out) -> void
    {
        int lane = threadIdx.x;
        assert_((0 <= lane) && (lane < 32));

        extern __shared__ u16 sm[];

        load_rot_8_11<dtype, 16, 8>(in, sm, lane);

        __syncwarp();

        for (int i = 0; i < 32; i++) {
            // 6..10
            u16* base_in = sm + 64 * i;
            u16* global_out = out + 64 * i;

            constexpr array<int, 10> strides = {
                1 << 0,  // 0: Id
                1 << 1,  // 1: Id
                1 << 2,  // 2:
                1 << 3,  // 3:
                1 << 4,  // 4: Id
                1 << 5,  // 5: Id
                1 << 11, // 11:
                1 << 12, // 12:
                1 << 13, // 13:
                1 << 14, // 14:
            };

            constexpr array<AxSpec, 10> spec = []() consteval {
                array<AxSpec, 10> spec = repeat_to_array<10>(AxSpec::Id);
                // required
                for (int i = 6; i < 10; i++) {
                    spec[i] = AxSpec::Rot;
                }
                // ones not taken by load_rot_8_11
                for (int ax : {3, 4, 5}) {
                    spec[ax] = AxSpec::Rot;
                }
                return spec;
            }();

            auto in_reg = Frag<
                dtype,
                10,
                AxSpec,
                spec,
                Perm<10>{
                    3,
                    4,
                    5,
                    6,
                    7,
                    //
                    0, // packing
                    8,
                    9,
                    1,
                    2,
                }
            >::template load<strides, SmBankCount<1>, 8>(base_in, lane);

            auto out_reg = hada_rot_8(in_reg, lane);
            static_assert(rot_finished(out_reg.AxMeta_));

            // Dummy<decltype(out_reg)>::x x;

            out_reg.template store<strides, SmBankCount<1>, 8>(global_out, lane);
        }
    }
};

template struct RowHandler<DType::Half, 15>;

// had dimension is 2**N
template <DType dtype, int N>
__global__ __launch_bounds__(32) auto hadamard_transform_ker(const u16* a, u16* out)
    -> void
{
    RowHandler<dtype, N> handler{};

    constexpr int R = handler.rows_per_block;
    constexpr int S = handler.log_elem_count;

    constexpr int elem_per_block = (1 << S) * R;

    // start pointers of what this block handles
    const u16* in_base = a + blockIdx.x * elem_per_block;
    u16* out_base = out + blockIdx.x * elem_per_block;

#pragma unroll
    for (int i = 0; i < R; i++) {

        const u16* a_i = in_base + i * (1 << S);
        u16* out_i = out_base + i * (1 << S);

        handler.handle_row(a_i, out_i);
    }
}

template <DType dtype, int N>
auto run_fht_helper(void* a_mat_ptr, void* out_ptr, uint32_t numel, cudaStream_t stream)
    -> void
{
    RowHandler<dtype, N> handler{};

    // # of elements to process in one block
    constexpr int elem_per_block = (1 << handler.log_elem_count) * handler.rows_per_block;

    TORCH_CHECK(numel % elem_per_block == 0);

    auto ker = hadamard_transform_ker<dtype, N>;
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(ker, cudaFuncAttributeMaxDynamicSharedMemorySize, handler.SM)
    );

    ker<<<numel / elem_per_block, 32, handler.SM, stream>>>(
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

            constexpr bool supported = (
                //
                // I == 8 || (8 < I && I <= 12) || (I == 15)
                I == 8 || I == 12 || I == 15
            );

            if constexpr (supported) {
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

// template auto run_fht<torch::ScalarType::BFloat16>(
//     void* a_mat_ptr,
//     void* out_ptr,
//     uint32_t numel,
//     uint32_t had_size,
//     cudaStream_t stream
// ) -> void;
