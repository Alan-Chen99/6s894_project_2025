#pragma once

#include "frag.cuh"

enum AxSpec {
    Id, // do nothing on the axis. it will effectively serve as a batch axis.
    Rot // apply a hadamard transform
};

template <size_t N> constexpr auto rot_finished(array<AxSpec, N> spec) -> bool
{
    for (auto x : spec) {
        if (x == AxSpec::Rot) {
            return false;
        }
    }
    return true;
}

template <
    float extra_scale = 1.0f,
    DType dtype,
    int N,
    array<AxSpec, N> spec,
    Perm<N * 2> P
>
__forceinline__ constexpr auto create_hada_A(int lane)
    -> Frag<dtype, N * 2, AxSpec, array_concat(spec, repeat_to_array<N>(AxSpec::Id)), P>
{
    return Frag<
        dtype,
        N * 2,
        AxSpec,
        array_concat(spec, repeat_to_array<N>(AxSpec::Id)),
        P
    >::create(lane, [](array<bool, N * 2> coord) -> u16 {
        // Count Rot axes
        constexpr int R = []() constexpr {
            int c = 0;
            for (int i = 0; i < N; ++i)
                if (spec[i] == AxSpec::Rot)
                    ++c;
            return c;
        }();

        static_assert(R > 0);

        // 2^(-R/2) = (0.5)^(R/2) * (1/sqrt(2))^(R%2)
        constexpr float scale_val = []() constexpr {
            float s = extra_scale;
            for (int i = 0; i < R; ++i)
                s *= INV_SQRT2;
            return s;
        }();

        constexpr u16 s_pos = f32_to_dtype<dtype>(scale_val);
        constexpr u16 s_neg = f32_to_dtype<dtype>(-scale_val);

        // typename Dummy2<scale_val>::x x;
        // typename Dummy2<s_neg>::x x;

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
    });
}

constexpr auto test_hada_a = create_hada_A<
    2.0f,
    DType::Half,
    4,
    array<AxSpec, 4>{AxSpec::Rot, AxSpec::Rot, AxSpec::Rot, AxSpec::Id},
    PermA
>(0);

constexpr auto test_collect(auto cb)
{
    array<decltype(cb(0)), 32> ans;
    for (int i = 0; i < 32; i++) {
        ans[i] = cb(i);
    }
    return ans;
}

template <float extra_scale = 1.0f, DType dtype, array<AxSpec, 7> spec>
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
    auto A = create_hada_A<extra_scale, dtype, 4, slice_array<4>(spec), PermA>(lane);
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
    float extra_scale = 1.0f,
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
        return hada_rot_4_impl<extra_scale>(frag, lane);
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
    float extra_scale = 1.0f,
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
        auto tmp1 = hada_rot_4<extra_scale>(frag, lane);

        // {0, 0, 1, 1, 1, 0, 0, 1}, {3, 4, 5, 0, 1, 2, 7, 6}
        auto tmp2 = tmp1.template transpose_layout<Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}>();

        // {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 7, 6}
        auto tmp3 = hada_rot_4(tmp2, lane);

        return tmp3.template transpose_layout<Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}>();
    });
}

template <size_t N>
constexpr auto hada_rot_axis_local_spec(array<AxSpec, N> spec, Perm<N> P, int K)
    -> array<AxSpec, N>
{
    assert(K >= 5 && K < N);
    assert(spec[P[K]] == AxSpec::Rot);

    array<AxSpec, N> ans = spec;
    // K is physical; update the corresponding logical axis spec to Id.
    ans[P[K]] = AxSpec::Id;
    return ans;
}

// Perform an unscaled 2-point Hadamard butterfly along physical axis K (K >= 5).
// For each pair (x0, x1) along that axis, compute:
//   y0 = x0 + x1
//   y1 = x0 - x1
// No memory ops; all in registers via add16/sub16.
// NOTE: No normalization is applied. Caller is responsible for any scaling.
template <int K, DType dtype, size_t N, array<AxSpec, N> spec, Perm<N> P>
__device__ __forceinline__ auto hada_rot_axis_local(Frag<dtype, N, AxSpec, spec, P> arr)
    -> Frag<dtype, N, AxSpec, hada_rot_axis_local_spec<N>(spec, P, K), P>
{
    static_assert(K >= 5 && K < N, "Axis must be a local (>=5) physical axis and < N");
    static_assert(K != 5, "TODO");

    array<u16, 1 << (N - 5)> out{};

    constexpr int L = 1 << (N - 5);      // elements per thread
    constexpr int stride = 1 << (K - 5); // pair distance along that axis
    constexpr int block = stride * 2;    // block size for the butterfly

#pragma unroll
    for (int base = 0; base < L; base += block) {
#pragma unroll
        for (int j = 0; j < stride; j += 2) {
            int i0 = base + j;
            int i1 = i0 + stride;

            u32 x0 = pack_b16x2(arr.data[i0], arr.data[i0 + 1]);
            u32 x1 = pack_b16x2(arr.data[i1], arr.data[i1 + 1]);

            u32 y0 = add16x2<dtype>(x0, x1);
            u32 y1 = sub16x2<dtype>(x0, x1);

            out[i0] = lo16(y0);
            out[i0 + 1] = hi16(y0);
            out[i1] = lo16(y1);
            out[i1 + 1] = hi16(y1);
        }
    }
    return {out};
}

// rotate all axis. requires N >= 8
template <
    float extra_scale = 1.0f,
    DType dtype,
    size_t N,
    array<AxSpec, N> spec,
    Perm<N> P
>
__device__ auto hada_rot_all(Frag<dtype, N, AxSpec, spec, P> arr, int lane)
    -> Frag<dtype, N, AxSpec, repeat_to_array<N>(AxSpec::Id), P>
{
    static_assert(N >= 8);

    constexpr int K_local = []() constexpr {
        for (int k = 8; k < N; ++k) {
            if (spec[P[k]] == AxSpec::Rot)
                return k;
        }
        return -1;
    }();

    if constexpr (K_local != -1) {
        return hada_rot_all<extra_scale * INV_SQRT2>(
            hada_rot_axis_local<K_local>(arr),
            lane
        );
    } else {
        return hada_rot_8<extra_scale>(arr, lane);
    }
}
