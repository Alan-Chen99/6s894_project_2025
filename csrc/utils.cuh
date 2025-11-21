#pragma once

#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

using u16 = uint16_t;
using u32 = uint32_t;

using std::array;
using std::pair;
using std::tuple;
using DType = torch::ScalarType;

////////////////////////////////////////

__host__ __device__ __forceinline__ void assume(bool cond)
{
#if defined(__CUDA_ARCH__)
    __builtin_assume(cond);
#endif
}

#define assert_(COND) \
    do { \
        bool _cond_eval = (COND); \
        assert(_cond_eval); \
        assume(_cond_eval); \
    } while (0)

template <typename T> struct Dummy {};
template <auto T> struct Dummy2 {};

////////////////////////////////////////

__device__ __forceinline__ void async_commit_group()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void async_wait_pending()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// copy 16 bytes between 16-byte aligned pointers
__device__ inline void cp_async16(void* smem_ptr, const void* glob_ptr)
{
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES)
    );
}

// copy 16 bytes
__device__ __forceinline__ void cp_16(void* dst, const void* src)
{
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
}

///////////////

template <DType dtype> struct CudaType;

template <> struct CudaType<DType::Half> {
    using Type = __nv_half;
};
template <> struct CudaType<DType::BFloat16> {
    using Type = __nv_bfloat16;
};

// ------------------- Small helpers (packing and bits) -----------------------

__device__ __forceinline__ u32 pack_b16x2(u16 lo, u16 hi)
{
    return static_cast<u32>(lo) | (static_cast<u32>(hi) << 16);
}
__device__ __forceinline__ u16 lo16(u32 x) { return static_cast<u16>(x & 0xFFFFu); }
__device__ __forceinline__ u16 hi16(u32 x) { return static_cast<u16>(x >> 16); }

template <std::size_t... Is, typename F>
__device__ void static_for_impl(std::index_sequence<Is...>, F&& f)
{
    (f.template operator()<Is>(), ...);
}

template <std::size_t N, typename F> __device__ void static_for(F&& f)
{
    static_for_impl(std::make_index_sequence<N>{}, std::forward<F>(f));
}

// template <DType dtype> __device__ __forceinline__ u16 add16(u16 x, u16 y)
// {
//     using T = CudaType<dtype>::Type;
//     T x_ = std::bit_cast<T>(x);
//     T y_ = std::bit_cast<T>(y);
//     T ans = x_ + y_;
//     return std::bit_cast<u16>(ans);
// }

// template <DType dtype> __device__ __forceinline__ u16 sub16(u16 x, u16 y)
// {
//     using T = CudaType<dtype>::Type;
//     T x_ = std::bit_cast<T>(x);
//     T y_ = std::bit_cast<T>(y);
//     T ans = x_ - y_;
//     return std::bit_cast<u16>(ans);
// }

// __device__ void testfn(u16 a, u16 b) { add16<DType::BFloat16>(a, b); }

// BlackBox is a clangd trick that otherwise does nothing.
//
// we wanted clangd to show full permutations on hover.
// sometimes ShowAKA config still dont work, so we use this hack
// clangd hover:
//
// no black box
// Type: OutType (aka Frag<8, Perm<8>{{{1, 0, 2, 3, 4, 5, 6, 7}}} + out_perm,
// (c10::ScalarType)'\x05'>)
//
// with black box
// Type: BlackBox<OutType> (aka Frag<8, Perm<8>{{{3, 4, 5, 1, 0, 2, 6, 7}}},
// c10::ScalarType::Half>)
//
// afaik showing full type through a error always works:
// Dummy<decltype(ans)>::x x;
// No type named ’x’ in ’Dummy<Frag<8, Perm<8>{{{3, 4, 5, 1, 0, 2, 6, 7}}},
// c10::ScalarType::Half>>’ [typename_nested_not_found]
template <typename T> struct BlackBox_ {
    using Type = T;
};
template <typename T> using BlackBox = BlackBox_<T>::Type;
