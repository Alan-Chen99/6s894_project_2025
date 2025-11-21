#pragma once

#include "frag.cuh"

template <DType dtype> constexpr u16 quarter_pos_bits()
{
    if constexpr (dtype == DType::Half)
        return static_cast<u16>(0x3400); // 0.25 in f16
    else
        return static_cast<u16>(0x3E80); // 0.25 in bf16
}
// -1/4
template <DType dtype> constexpr u16 quarter_neg_bits()
{
    return static_cast<u16>(quarter_pos_bits<dtype>() | 0x8000u);
}

template <DType dtype, int N, Perm<N * 2> P>
__forceinline__ constexpr auto create_hada_A(int lane) -> Frag<dtype, N * 2, P>
{
    return Frag<dtype, N * 2, P>::create(
        lane, //
        [](array<bool, N * 2> coord) -> u16 {
            int ans = 0;
            for (int i = 0; i < N; i++) {
                if (coord[i] && coord[i + N]) {
                    ans++;
                }
            }

            constexpr u16 qpos = quarter_pos_bits<dtype>();
            constexpr u16 qneg = quarter_neg_bits<dtype>();

            if (ans % 2 == 0) {
                return qpos;
            } else {
                return qneg;
            }
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
template <DType dtype, int N, Perm<N> P>
__device__ auto hada_rot_4(Frag<dtype, N, P> arr, int lane)
    -> BlackBox<Frag<dtype, N, P + Perm<7>{3, 4, 5, 0, 1, 2, 6}>>
{
    auto ans = apply_transposed<7, PermB>(arr, [&](Frag<dtype, 7, PermB> frag) {
        auto A = create_hada_A<dtype, 4, PermA>(lane);

        // Had rot on the first 4 logical axis of frag
        return mma_m16_n8_k16<dtype>(A, frag);
    });

    // we logically transposed id({0, 1, 2, 3, 4, 5, 6}) -> PermB({3, 4, 5, 0, 1, 2, 6})
    // so we actually rotated {0, 1, 5, 6}
    return ans;
}

// rotate first 8 axis: P[0], P[1], ...
//
template <DType dtype, int N, Perm<N> P>
__device__ auto hada_rot_8(Frag<dtype, N, P> arr, int lane) -> Frag<dtype, N, P>
{
    auto ans = apply_transposed<8>(arr, [&](Frag<dtype, 8> frag) {
        // /**/ marked axis are rotated
        Frag<dtype, 8, Perm<8>{3, 4, 5 /**/, 0 /**/, 1 /**/, 2, 6 /**/, 7}> tmp1 =
            // roates position [0, 1, 5, 6]
            hada_rot_4(frag, lane);

        Frag<dtype, 8, Perm<8>{3, 4, 5 /**/, 0 /**/, 1 /**/, 2, 7, 6 /**/}> tmp2 =
            tmp1.template transpose_layout<Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}>();

        // all is rotated now
        Frag<dtype, 8, Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}> tmp3 = hada_rot_4(tmp2, lane);

        // not mandatory but its free to make it identity
        return tmp3.template transpose_layout<Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}>();
    });

    return ans;
}

__device__ void testfn(Frag<DType::BFloat16, 9> arr)
{
    auto blah = hada_rot_4(arr, 0);

    // constexpr auto ans = create_hada_A<DType::Half, 4, PermA>(0);

    // constexpr auto test = test_collect([](int lane) {
    //     return create_hada_A<DType::Half, 4, PermA>(lane).data;
    // });

    // // Dummy2<test>::x x;
}
