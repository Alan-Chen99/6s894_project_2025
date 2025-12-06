#pragma once

#include "frag.cuh"
#include "hada.cuh"
#include "utils.cuh"

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

    for (int I = 0; I < N + P; I++) {
        if (I < N) {
            cp_async16(
                sm + I * S1 + lane * 8, // shared: stride S
                in + I * 256 + lane * 8 // gobal: stride 256
            );
            async_commit_group();
        }

        if (P <= I) {
            int idx = I - P;
            int ongoing = std::min(I + 1, N);

            async_wait_pending_dyn<P>(ongoing - idx - 1);
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
    }
}

// rotate 0..9 except 3, 4
template <
    DType dtype,
    int N,
    int P // # of cp.async pipeline at once
>
__device__ auto load_rot_8_11(const u16* in, u16* sm, int lane) -> void
{
    for (int I = 0; I < N + P; I++) {
        if (I < N) {
            // copy 2048 contiguous elements
            for (int i = 0; i < 8; i++) {
                int offset = I * 2048 + 256 * i + lane * 8;
                cp_async16(sm + offset, in + offset);
            }
            async_commit_group();
        }

        if (P <= I) {
            int idx = I - P;
            int ongoing = std::min(I + 1, N);

            async_wait_pending_dyn<P>(ongoing - idx - 1);
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
    }
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

    static constexpr int SM_BYTES = 0;

    __device__ static auto handle_row(const u16* in, u16* out, int lane, void* sm) -> void
    {
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

    static constexpr int S = 256 + 8;
    static constexpr int SM_BYTES = S * 16 * 2;

    __device__ static auto handle_row(const u16* in, u16* out, int lane, void* sm_)
        -> void
    {
        u16* sm = static_cast<u16*>(sm_);

        load_rot_8<dtype, 16, 7, S>(in, sm, lane);

        __syncwarp();

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
    requires(12 < N && N <= 15)
// requires(N == 15)
struct RowHandler<dtype, N> {
    static constexpr int log_elem_count = 15;
    static constexpr int rows_per_block = 1;

    static constexpr int SM_BYTES = (1 << 15) * 2;

    __device__ static auto handle_row(const u16* in, u16* out, int lane, void* sm_)
        -> void
    {
        u16* sm = static_cast<u16*>(sm_);

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
                for (int i = 6; i < N - 5; i++) {
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

constexpr auto size_supported(int N) -> bool
{
    // return N == 8 || (8 < N && N <= 12) || (12 < N && N <= 15);
    return 8 <= N && N <= 15;
}

/**
 * Performs a specific length Fast Hadamard Transform on the device.
 *
 * It is designed to be called by a warp (32 threads) within a CUDA kernel.
 *
 * Executes synchronously on the calling warp.
 */
template <
    DType dtype,
    int N // Log2 of the length of a *single* transform vector
>
struct HadamardTransform {
    using _Handler = RowHandler<dtype, N>;

    // The total number of consecutive elements processed in one call.
    // I will be reducing this for some sizes
    static constexpr int T = 1 << _Handler::log_elem_count;

    // Number of rows (vectors of length 2^N) processed in a single batch
    static constexpr int N_ROWS = T / (1 << N);

    // amount of shared memory needed
    // I will be reducing this for some sizes
    static constexpr int SM_BYTES = _Handler::SM_BYTES;

    __device__ static auto hadamard_transform_device(
        const u16* in, // Pointer to the input data in global memory
        u16* out,      // Pointer to the output location in global memory
        void* sm       // shared memory pointer, SM_BYTES bytes needed
    )
    {
        int lane = threadIdx.x % 32;
        assert_((0 <= lane) && (lane < 32));

        _Handler handler;
        handler.handle_row(in, out, lane, sm);
    }
};

// to get clangd diagnostics
inline void inst_templates()
{
    static_for<16>([&]<int N>() {
        if constexpr (size_supported(N)) {
            auto F1 = RowHandler<DType::Half, N>::handle_row;
            auto F2 = RowHandler<DType::BFloat16, N>::handle_row;
        }
    });
}
