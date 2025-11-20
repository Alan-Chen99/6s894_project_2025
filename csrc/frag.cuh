#pragma once

#include "utils.cuh"
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

template <int N> struct Perm {
    array<int, N> ord;

    // Identity permutation of size N
    constexpr static auto ID() -> Perm<N>
    {
        array<int, N> ord{};
        for (int i = 0; i < N; i++)
            ord[i] = i;
        return {ord};
    }

    constexpr auto size() -> int { return ord.size(); }
    constexpr auto operator[](int idx) const -> int { return ord[idx]; }

    // adding represent a added premutation to a fragment
    // a function will have signature of this form:
    //
    // template<Perm> auto func(Frag<Perm>) -> Frag<Perm + {2, 1, 0}>
    template <int M>
    constexpr auto operator+(const Perm<M>& p2) const -> Perm<(N > M ? N : M)>
    {
        constexpr int R = (N > M ? N : M);
        Perm<R> out{};
        for (int i = 0; i < R; ++i) {
            const int a = (i < M) ? p2[i] : i;
            out.ord[i] = (a < N) ? ord[a] : a;
        }
        return out;
    }
};

// for shared mem, with 32 bank model
// how many loads does it take?
template <int limit = 1> struct SmBankCount {
    // offsets in words
    static constexpr auto check(array<int, 32> word_offsets)
    {
        array<int, 32> banks{};
        for (int offset : word_offsets) {
            banks[(offset % 32 + 32) % 32] += 1;
        }

        int mx = 0;

        for (int count : banks) {
            mx = std::max(mx, count);
        }

        // can be relaxed to <= if desirable
        if (mx == limit) {
            return true;
        } else {
            return false;
        }
    }
};

// more strict than need currently
struct Coalesced {
    static constexpr auto check(array<int, 32> word_offsets)
    {
        for (int i = 0; i < 32; i++) {
            if (word_offsets[i] != i) {
                return false;
            }
        }
        return true;
    };
};

struct OK {
    using get = void;
};

template <typename T> struct CHECK {
    using fail = T::do_fail;
};
template <> struct CHECK<OK> {
    using OK = void;
};

template <auto V> using CHECK_V = CHECK<decltype(V)>::OK;

template <auto Val, typename MEM> struct MemCheckFailed {};

template <int N> constexpr auto default_strides() -> array<int, N>
{
    array<int, N> ans{};
    for (int i = 0; i < N; i++)
        ans[i] = 1 << i;
    return ans;
}

template <int N> constexpr auto find_item(array<int, N> arr, int item) -> int
{
    for (int i = 0; i < N; i++) {
        if (arr[i] == item) {
            return i;
        }
    }
    assert(false);
    return 0;
}

// try to pack a bunch of u16 loads to u32 loads
template <int N>
constexpr auto compute_pack(array<int, N> offsets)
    -> tuple<array<int, N / 2>, array<int, N / 2>, array<int, N / 2>>
{
    array<int, N / 2> offsets_load{};
    int cur = 0;
    for (int offset : offsets) {
        if (offset % 2 == 0) {
            offsets_load[cur] = offset;
            cur++;
        }
    }
    assert(cur == N / 2);

    array<int, N / 2> idxs_even{};
    array<int, N / 2> idxs_odd{};

    for (int i = 0; i < N; i++) {
        if (offsets[i] % 2 == 0) {
            auto item = find_item<N / 2>(offsets_load, offsets[i]);
            idxs_even[item] = i;
        } else {
            auto item = find_item<N / 2>(offsets_load, offsets[i] - 1);
            idxs_odd[item] = i;
        }
    }
    return {offsets_load, idxs_even, idxs_odd};
}

template <int N, array<int, N> offsets>
__device__ __forceinline__ auto load_offsets_aligned(const u16* base) -> array<u16, N>
{
    constexpr auto offsets_packed = compute_pack<N>(offsets);
    auto [offsets_load, idxs_even, idxs_odd] = offsets_packed;

    array<u16, N> ans;
    const u32* base_ = reinterpret_cast<const u32*>(base);

    for (int i = 0; i < N / 2; i++) {
        auto tmp = base_[offsets_load[i] / 2];
        ans[idxs_even[i]] = lo16(tmp);
        ans[idxs_odd[i]] = hi16(tmp);
    }

    return ans;
}

template <int N, array<int, N> offsets>
__device__ __forceinline__ auto store_offsets_aligned(u16* base, array<u16, N> data)
    -> void
{
    constexpr auto offsets_packed = compute_pack<N>(offsets);
    auto [offsets_load, idxs_even, idxs_odd] = offsets_packed;

    u32* base_ = reinterpret_cast<u32*>(base);

    for (int i = 0; i < N / 2; i++) {
        auto val = pack_b16x2(data[idxs_even[i]], data[idxs_odd[i]]);
        base_[offsets_load[i] / 2] = val;
    }
}

constexpr auto is_linear(array<int, 32> offsets) -> bool
{
    int base = offsets[0];
    int factor = offsets[1];
    for (int i = 0; i < 32; i++) {
        if (offsets[i] != base + factor * i) {
            return false;
        }
    }
    return true;
}

constexpr auto u16_to_u32_offsets(array<int, 32> offsets) -> array<int, 32>
{
    array<int, 32> ans;
    for (int i = 0; i < 32; i++) {
        assert(offsets[i] % 2 == 0);
        ans[i] = offsets[i] / 2;
    }
    return ans;
}

// ------------------------- FRAGMENT view (Frag) -----------------------------
//
// a nd array of shape 2x2x...x2, sharded across a warp
//
// each u16 has a lane index `l` and a data array index `i`
//
// l0, the least sig bit of l, represent the logical index on logical axis P[0]
// l1, represent the logical index on logical axis P[1]
// etc
//
// i0, the least sig bit of i, represent the logical index on logical axis P[5]
// i1, represent the logical index on logical axis P[6]
// etc
//
template <DType dtype, int N, Perm<N> P = Perm<N>::ID()> struct Frag {
    array<u16, 1 << (N - 5)> data;

    static constexpr auto perm() -> Perm<N> { return P; }

    // suppose the frag is in memory.
    //   strides: stride of logical axis
    // get the offset each lane needs to apply
    static constexpr auto get_lane_offset_impl(array<int, N> strides, int lane) -> int
    {
        int ans = 0;
        for (int i = 0; i < 5; i++)
            if (lane & (1 << i))
                ans += strides[P[i]];
        return ans;
    }

    static constexpr auto get_lane_offset_all(array<int, N> strides) -> array<int, 32>
    {
        array<int, 32> ans;
        for (int i = 0; i < 32; i++) {
            ans[i] = get_lane_offset_impl(strides, i);
        }
        return ans;
    }

    template <array<int, N> strides, typename MEM>
    static constexpr auto mem_check_strides()
    {
        constexpr auto offsets = get_lane_offset_all(strides);

        if constexpr (MEM::check(u16_to_u32_offsets(offsets))) {
            return OK();
        } else {
            return MemCheckFailed<strides, MEM>();
        }
    }

    // get_lane_offset_impl but optentially optimized and checked with mem_check_strides
    template <
        array<int, N> strides,
        typename MEM,
        typename = CHECK_V<mem_check_strides<strides, MEM>()>
    >
    __device__ __forceinline__ static auto get_lane_offset(int lane) -> int
    {
        assert_((0 <= lane) && (lane < 32));

        constexpr auto offsets = get_lane_offset_all(strides);

        if constexpr (is_linear(offsets)) {
            // fast path
            constexpr int base = offsets[0];
            constexpr int factor = offsets[1] - offsets[0];

            return base + factor * lane;

        } else {
            return get_lane_offset_impl(strides, lane);
        }
    }

    static constexpr auto is_simple_trans(Perm<N> Trans) -> bool
    {
        for (int i = 0; i < 5; i++) {
            if (Trans[i] != i) {
                return false;
            }
        }
        return true;
    }

    // transpose layout without changing logical array content
    // currently only among local dimension is supported
    template <Perm<N> Trans>
    __device__ auto transpose_layout() -> Frag<dtype, N, P + Trans>
    {
        static_assert(is_simple_trans(Trans));

        using Out = Frag<dtype, N, P + Trans>;
        Out out{};

        constexpr int L = 1 << (N - 5);
        if constexpr (L == 1) {
            out.data[0] = data[0];
            return out;
        }

        // Invert Trans over full N (cheap); we only use the local part.
        array<int, N> inv{};
        for (int i = 0; i < N; ++i)
            inv[Trans[i]] = i;

        // For each old local index, compute its destination index by permuting
        // bit-positions.
        for (int src = 0; src < L; ++src) {
            int dst = 0;
            for (int k = 0; k < (N - 5); ++k) {
                if (src & (1 << k)) {
                    int k_new = inv[5 + k] - 5; // where old bit k lands
                    dst |= (1 << k_new);
                }
            }
            out.data[dst] = data[src];
        }
        return out;
    }

    static constexpr auto get_local_offsets(array<int, N> strides)
        -> array<int, 1 << (N - 5)>
    {
        array<int, 1 << (N - 5)> ans{};
        for (int i = 0; i < ans.size(); i++) {
            int ofs = 0;
            for (int j = 0; j < N - 5; j++)
                if (i & (1 << j))
                    ofs += strides[P[j + 5]];
            ans[i] = ofs;
        }
        return ans;
    }

    template <
        array<int, N> strides = default_strides<N>(), // logical strides
        typename MEM = SmBankCount<1>,
        typename = CHECK_V<mem_check_strides<strides, MEM>()>
    >
    __device__ static auto load(const u16* ptr, int lane) -> Frag<dtype, N, P>
    {
        using F = Frag<dtype, N, P>;

        ptr += F::template get_lane_offset<strides, MEM>(lane);
        constexpr auto offsets = F::get_local_offsets(strides);

        array<u16, 1 << (N - 5)> ans = load_offsets_aligned<offsets.size(), offsets>(ptr);
        return {ans};
    }

    template <
        array<int, N> strides = default_strides<N>(), // logical strides
        typename MEM = SmBankCount<1>,
        typename = CHECK_V<mem_check_strides<strides, MEM>()>
    >
    __device__ auto store(u16* ptr, int lane) const
    {
        ptr += get_lane_offset<strides, MEM>(lane);
        constexpr auto offsets = get_local_offsets(strides);

        store_offsets_aligned<offsets.size(), offsets>(ptr, data);
    }
};
