#pragma once

#include "frag.cuh"

enum AxSpec {
    Id, // do nothing on the axis. it will effectively serve as a batch axis.
    Rot // apply a hadamard transform
};

template <DType dtype, int N, array<AxSpec, N> spec, Perm<N * 2> P>
__forceinline__ constexpr auto create_hada_A(int lane)
    -> Frag<dtype, N * 2, AxSpec, array_concat(spec, repeat_to_array<N>(AxSpec::Id)), P>
{
    return Frag<
        dtype,
        N * 2,
        AxSpec,
        array_concat(spec, repeat_to_array<N>(AxSpec::Id)),
        P
    >::
        create(
            lane, //
            [](array<bool, N * 2> coord) -> u16 {
                // Count Rot axes
                constexpr int R = []() constexpr {
                    int c = 0;
                    for (int i = 0; i < N; ++i)
                        if (spec[i] == AxSpec::Rot)
                            ++c;
                    return c;
                }();

                // 2^(-R/2) = (0.5)^(R/2) * (1/sqrt(2))^(R%2)
                constexpr float inv_sqrt2 = 0.7071067811865475244f;
                constexpr float scale_val = []() constexpr {
                    float s = 1.0f;
                    for (int i = 0; i < (R / 2); ++i)
                        s *= 0.5f;
                    if (R & 1)
                        s *= inv_sqrt2;
                    return s;
                }();
                constexpr u16 s_pos = f32_to_dtype<dtype>(scale_val);
                constexpr u16 s_neg = f32_to_dtype<dtype>(-scale_val);

                // coord[0..N-1] = k bits, coord[N..2N-1] = m bits
                // Id axes: Kronecker delta (k_i == m_i), otherwise 0
                // Rot axes: contribute (-1)^{k_i & m_i} to the sign
                int parity = 0;
                for (int i = 0; i < N; ++i) {
                    bool ki = coord[i];
                    bool mi = coord[i + N];
                    if (spec[i] == AxSpec::Id) {
                        if (ki != mi) {
                            return static_cast<u16>(0);
                        }
                    } else { // AxSpec::Rot
                        if (ki && mi) {
                            parity ^= 1;
                        }
                    }
                }
                return (parity == 0) ? s_pos : s_neg;
            }
        );
}

constexpr auto test_collect(auto cb)
{
    array<decltype(cb(0)), 32> ans;
    for (int i = 0; i < 32; i++) {
        ans[i] = cb(i);
    }
    return ans;
}

template <DType dtype, array<AxSpec, 7> spec>
__device__ auto hada_rot_4_impl(Frag<dtype, 7, AxSpec, spec, PermB> arr, int lane)
    -> Frag<
        dtype,
        7,
        AxSpec,
        array<
            AxSpec,
            7
        >{AxSpec::Id, AxSpec::Id, AxSpec::Id, AxSpec::Id, spec[4], spec[5], spec[6]},
        PermC
    >
{
    auto A = create_hada_A<dtype, 4, slice_array<4>(spec), PermA>(lane);
    // Had rot on the first 4 logical axis
    return mma_m16_n8_k16<dtype>(A, arr);
}

template <size_t N>
constexpr auto hada_rot_4_axis(array<AxSpec, N> spec, Perm<N> P) -> array<AxSpec, N>
{
    array<AxSpec, N> ans = spec;
    for (auto x : {0, 1, 5, 6}) {
        ans[P[x]] = AxSpec::Id;
    }
    return ans;
}

// <logical behavior>
//
// Hadamard rotate axis P[0], P[1], P[5], P[6]
// The output will have a different axis spec.
//
// arr must be at least 7 dimenstional.
//
// <physical behavior>
//
// apply rotation passing first 7 axis to tensor core,
// and using the rest as batch dim
//
template <
    //
    DType dtype,
    size_t N,
    array<AxSpec, N> spec,
    Perm<N> P
>
__device__ auto hada_rot_4(Frag<dtype, N, AxSpec, spec, P> arr, int lane) -> BlackBox<
    Frag<dtype, N, AxSpec, hada_rot_4_axis(spec, P), P + Perm<7>{3, 4, 5, 0, 1, 2, 6}>
>
{
    auto ans = arr.template apply_transposed<7, PermB>([&](auto frag) {
        return hada_rot_4_impl(frag, lane);
    });
    return ans;
}

template <size_t N>
constexpr auto hada_rot_8_axis(array<AxSpec, N> spec, Perm<N> P) -> array<AxSpec, N>
{
    array<AxSpec, N> ans = spec;
    for (int i = 0; i < 8; i++) {
        ans[P[i]] = AxSpec::Id;
    }
    return ans;
}

// rotate first 8 axis: P[0], P[1], ...
template <
    //
    DType dtype,
    size_t N,
    array<AxSpec, N> spec,
    Perm<N> P
>
__device__ auto hada_rot_8(Frag<dtype, N, AxSpec, spec, P> arr, int lane)
    -> Frag<dtype, N, AxSpec, hada_rot_8_axis(spec, P), P>
{
    return arr.template apply_transposed<8>([&](auto frag) {
        // {0, 0, 1, 1, 1, 0, 0, 1}, {3, 4, 5, 0, 1, 2, 6, 7}
        auto tmp1 = hada_rot_4(frag, lane);

        // {0, 0, 1, 1, 1, 0, 0, 1}, {3, 4, 5, 0, 1, 2, 7, 6}
        auto tmp2 = tmp1.template transpose_layout<Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}>();

        // {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 7, 6}
        auto tmp3 = hada_rot_4(tmp2, lane);

        return tmp3.template transpose_layout<Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}>();
    });
}
