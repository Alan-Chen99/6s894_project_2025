#pragma once

#include "dtype.h"
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>

using u16 = uint16_t;
using u32 = uint32_t;

using std::array;
using std::pair;
using std::tuple;

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

// TODO: seems to generate extra asm and does nothing better
template <int align, typename T>
__host__ __device__ __forceinline__ T* assume_aligned_ptr(T* p)
{
#if defined(__CUDA_ARCH__)
    return reinterpret_cast<T*>(__builtin_assume_aligned(p, align));
#else
    return p;
#endif
}

// result needs to be written back to pointer to be effective
#define assert_aligned(PTR, ALIGN) \
    (assert((((size_t)(ALIGN)) & (((size_t)(ALIGN)) - 1)) == 0), /* power-of-two */ \
     assert((reinterpret_cast<uintptr_t>(PTR) & (((size_t)(ALIGN)) - 1)) == 0), \
     /* assume_aligned_ptr<ALIGN>(PTR) */ (PTR))

template <typename T> struct Dummy {};
template <auto T> struct Dummy2 {};

constexpr float INV_SQRT2 = 0.7071067811865475244f;

////////////////////////////////////////

template <typename T> constexpr auto div_ceil(T x, T y) -> T { return (x + y - 1) / y; }

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
__host__ __device__ void static_for_impl(std::index_sequence<Is...>, F&& f)
{
    (f.template operator()<Is>(), ...);
}

template <std::size_t N, typename F> __host__ __device__ void static_for(F&& f)
{
    static_for_impl(std::make_index_sequence<N>{}, std::forward<F>(f));
}

template <DType dtype> __device__ __forceinline__ u16 add16(u16 x, u16 y)
{
    using T = CudaType<dtype>::Type;
    T x_ = std::bit_cast<T>(x);
    T y_ = std::bit_cast<T>(y);
    T ans = __hadd(x_, y_);
    return std::bit_cast<u16>(ans);
}

template <DType dtype> __device__ __forceinline__ u16 sub16(u16 x, u16 y)
{
    using T = CudaType<dtype>::Type;
    T x_ = std::bit_cast<T>(x);
    T y_ = std::bit_cast<T>(y);
    T ans = __hsub(x_, y_);
    return std::bit_cast<u16>(ans);
}

// Pairwise add of two packed scalars (lo, hi) in one 32-bit register.
// Returns elementwise sum in packed form.
// For fp16: add.f16x2
// For bf16: add.rn.bf16x2 (requires sm_80+)
template <DType dtype> __device__ __forceinline__ u32 add16x2(u32 a, u32 b)
{
    u32 r;
    if constexpr (dtype == DType::Half) {
        asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    } else {
        // bfloat16
        asm volatile("add.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    }
    return r;
}

// Packed subtract: (lo,hi) = (a.lo - b.lo, a.hi - b.hi)
template <DType dtype> __device__ __forceinline__ u32 sub16x2(u32 a, u32 b)
{
    u32 r;
    if constexpr (dtype == DType::Half) {
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    } else {
        asm volatile("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    }
    return r;
}

////////////////////////////////////////////////////////////
// LLM generated function to constexpr convert f32 to f16/bf16

// IEEE-754 float32 -> float16 (binary16) with round-to-nearest-even
__forceinline__ constexpr u16 f16_from_f32_bits(u32 f)
{
    u32 sign = (f >> 16) & 0x8000u; // move sign to bit 15
    u32 exp = (f >> 23) & 0xFFu;
    u32 mant = f & 0x7FFFFFu;

    // NaN/Inf
    if (exp == 0xFFu) {
        if (mant == 0)
            return static_cast<u16>(sign | 0x7C00u); // inf
        // qNaN: preserve payload as much as possible, ensure at least one mant bit
        u16 payload = static_cast<u16>(mant >> 13);
        if (payload == 0)
            payload = 1;
        return static_cast<u16>(sign | 0x7C00u | payload);
    }

    // Compute unbiased exponent for half
    int e = static_cast<int>(exp) - 127 + 15;

    // Underflow to subnormal or zero
    if (e <= 0) {
        if (e <= -10) {
            // Too small, becomes signed zero
            return static_cast<u16>(sign);
        }
        // Subnormal: restore implicit leading 1 then shift with RN-even
        mant |= 0x00800000u; // add hidden 1
        int shift = 14 - e;  // (1 - e) + (23 - 10)
        u32 trunc = mant >> shift;
        u32 rem = mant & ((1u << shift) - 1u);
        u32 half = 1u << (shift - 1);
        u32 inc = (rem > half) || (rem == half && (trunc & 1u));
        u32 mant10 = trunc + inc;
        return static_cast<u16>(sign | mant10); // exp=0
    }

    // Overflow -> Inf
    if (e >= 31) {
        return static_cast<u16>(sign | 0x7C00u);
    }

    // Normalized case: round mantissa from 23 to 10 bits (RN-even)
    u32 trunc = mant >> 13;
    u32 rem = mant & 0x1FFFu;
    u32 inc = (rem > 0x1000u) || (rem == 0x1000u && (trunc & 1u));
    trunc += inc;

    // Mantissa overflow bumps exponent
    if (trunc == 0x400u) {
        ++e;
        trunc = 0;
        if (e >= 31) {
            return static_cast<u16>(sign | 0x7C00u);
        }
    }

    return static_cast<u16>(sign | (static_cast<u32>(e) << 10) | trunc);
}

__forceinline__ constexpr u16 f16_from_f32(float x)
{
    u32 bits = std::bit_cast<u32>(x);
    return f16_from_f32_bits(bits);
}

/////

__forceinline__ constexpr u16 bf16_from_f32_bits(u32 fbits)
{
    u16 top = static_cast<u16>(fbits >> 16);
    u32 lsb = static_cast<u32>(top & 1u);
    u32 rounded = fbits + 0x7FFFu + lsb; // RN-even
    return static_cast<u16>(rounded >> 16);
}

__forceinline__ constexpr u16 bf16_from_f32(float x)
{
    u32 bits = std::bit_cast<u32>(x);
    return bf16_from_f32_bits(bits);
}

/////

template <DType dtype> constexpr u16 f32_to_dtype(float x)
{
    if constexpr (dtype == DType::Half) {
        return f16_from_f32(x);
    } else {
        static_assert(dtype == DType::BFloat16);
        return bf16_from_f32(x);
    }
}

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

template <int N> constexpr auto repeat_to_array(auto x) -> array<decltype(x), N>
{
    array<decltype(x), N> ans;
    for (int i = 0; i < N; i++) {
        ans[i] = x;
    }
    return ans;
}

template <size_t M, typename T, size_t N>
constexpr auto slice_array(array<T, N> arr) -> array<T, M>
{
    static_assert(M <= N);
    array<T, M> ans;
    for (size_t i = 0; i < M; i++) {
        ans[i] = arr[i];
    }
    return ans;
}

template <typename T, size_t N, size_t M>
constexpr auto array_concat(array<T, N> a1, array<T, M> a2) -> array<T, N + M>
{
    array<T, N + M> ans;
    for (int i = 0; i < N; i++) {
        ans[i] = a1[i];
    }
    for (int i = 0; i < M; i++) {
        ans[N + i] = a2[i];
    }
    return ans;
}
