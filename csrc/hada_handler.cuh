#pragma once

#include "frag.cuh"
#include "hada.cuh"
#include "utils.cuh"

/////////////////////////////////////////////////

template <
    DType dtype,
    int N,
    array<AxSpec, N> spec,
    Perm<N> perm, // register layout. does not effect result.
    int COUNT,
    int P, // # of cp.async pipeline at once
    float extra_scale = 1.0f
>
__device__ auto load_rot_n(const u16* in, u16* sm, int lane) -> void
{
    static_assert(N >= 8);

    for (int I = 0; I < COUNT + P; I++) {
        if (I < COUNT) {
            // copy 1<<N contiguous elements
            for (int i = 0; i < (1 << (N - 8)); i++) {
                int offset = I * (1 << N) + 256 * i + lane * 8;
                cp_async16(sm + offset, in + offset);
            }
            async_commit_group();
        }

        if (P <= I) {
            int idx = I - P;
            int ongoing = std::min(I + 1, COUNT);

            async_wait_pending_dyn<P>(ongoing - idx - 1);
            __syncwarp();

            u16* data = sm + idx * (1 << N);

            auto in_reg = Frag<
                dtype,
                N,
                AxSpec,
                spec,
                perm
            >::
                template load<
                    default_strides<N>(), //
                    SmBankCount<1>,
                    8
                >(data, lane);

            auto out_reg = hada_rot_all<extra_scale>(in_reg, lane);
            // typename Dummy<decltype(out_reg)>::x x;

            static_assert(rot_finished(out_reg.AxMeta_));

            out_reg.template store<
                default_strides<N>(), //
                SmBankCount<1>,
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

template <DType dtype, int N>
    requires(8 <= N && N <= 11)
struct RowHandler<dtype, N> {
    static constexpr int log_elem_count = N;
    static constexpr int rows_per_block = 1 << std::max(0, (10 - N));

    static constexpr int SM_BYTES = 0;

    __device__ static auto handle_row(const u16* in, u16* out, int lane, void* sm) -> void
    {
        static_assert(N >= 8);
        constexpr Perm<N> perm = []() consteval {
            array<int, N> ans = {3, 4, 5, 6, 7, /**/ 0, 1, 2};
            for (int i = 8; i < N; i++) {
                ans[i] = i;
            }
            return Perm<N>{ans};
        }();

        constexpr auto strides = default_strides<N>();

        auto in_reg = Frag<dtype, N, AxSpec, repeat_to_array<N>(AxSpec::Rot), perm>::
            template load<strides, Coalesced, 8>(in, lane);

        auto out_reg = hada_rot_all(in_reg, lane);
        static_assert(rot_finished(out_reg.AxMeta_));

        out_reg.template store<strides, Coalesced, 8>(out, lane);
    }
};

// 14 -> [6+4], [6+4]
template <DType dtype, int N>
    requires(12 <= N && N <= 15)
struct RowHandler<dtype, N> {
    static constexpr int log_elem_count = N;
    static constexpr int rows_per_block = 1;

    static constexpr int SM_BYTES = (1 << N) * 2;

    __device__ static auto handle_row(const u16* in, u16* out, int lane, void* sm_)
        -> void
    {
        u16* sm = static_cast<u16*>(sm_);

        // 15 -> [6+5], [6+4]
        // 14 -> [6+4], [6+4]
        // 13 -> [6+4], [6+3]
        // 12 -> [6+3], [6+3]
        constexpr int N1 = (N + 1) / 2 + 3;
        constexpr int N2 = N / 2 + 3;
        static_assert(N1 + N2 == N + 6);

        /////

        constexpr Perm<N1> perm1 = []() consteval -> Perm<N1> {
            array<int, N1> ans = {3, 4, 5, 6, 7, /**/ 0};
            for (int i = 8; i < N1; i++) {
                ans[i - 2] = i;
            }
            ans[N1 - 2] = 1;
            ans[N1 - 1] = 2;
            return {ans};
        }();

        // 3, 4, 5, 6, 7, 0, 8, 9, 10, 1, 2
        // Dummy2<perm1>::x x;

        constexpr Perm<N2> perm2 = []() consteval -> Perm<N2> {
            array<int, N2> ans = {3, 4, 5, 6, 7, /**/ 0};
            for (int i = 8; i < N2; i++) {
                ans[i - 2] = i;
            }
            ans[N2 - 2] = 1;
            ans[N2 - 1] = 2;
            return {ans};
        }();

        // 3, 4, 5, 6, 7, 0, 8, 9, 1, 2
        // Dummy2<perm2>::x x;

        /////

        // second phase will be a single rot_8
        static_assert(N2 <= 10);
        constexpr array<AxSpec, N2> spec2 = [=]() consteval {
            array<AxSpec, N2> ans = repeat_to_array<N2>(AxSpec::Id);
            for (int i = 6; i < N2; i++) {
                ans[i] = AxSpec::Rot;
            }
            for (int i = 0; i < 8; i++) {
                ans[perm2[i]] = AxSpec::Rot;
            }
            return ans;
        }();

        // everything not handled by second phase
        constexpr array<AxSpec, N1> spec1 = [=]() consteval {
            array<AxSpec, N1> ans = repeat_to_array<N1>(AxSpec::Rot);
            for (int i = 0; i < 6; i++) {
                if (spec2[i] == AxSpec::Rot) {
                    ans[i] = AxSpec::Id;
                }
            }
            return ans;
        }();

        // Dummy2<spec1>::x x;
        //////

        load_rot_n<dtype, N1, spec1, perm1, 1 << (N - N1), 7>(in, sm, lane);
        __syncwarp();

        for (int i = 0; i < (1 << (N - N2)); i++) {
            u16* base_in = sm + 64 * i;
            u16* global_out = out + 64 * i;

            // everything not handled by second phase
            constexpr array<int, N2> strides = [=]() consteval {
                array<int, N2> strides;
                for (int i = 0; i < 6; i++) {
                    strides[i] = 1 << i;
                }
                for (int i = 6; i < N2; i++) {
                    strides[i] = 1 << (i + (N1 - 6));
                }
                return strides;
            }();
            // Dummy2<strides>::x x;

            auto in_reg = Frag<dtype, N2, AxSpec, spec2, perm2>::
                template load<strides, SmBankCount<1>, 8>(base_in, lane);

            auto out_reg = hada_rot_8(in_reg, lane);

            static_assert(rot_finished(out_reg.AxMeta_));

            // Dummy<decltype(out_reg)>::x x;
            out_reg.template store<strides, Coalesced, 8>(global_out, lane);
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
