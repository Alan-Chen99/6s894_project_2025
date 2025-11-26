#pragma once

#include "utils.cuh"
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

template <size_t N> struct Perm {
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

    constexpr auto inv() const -> Perm<N>
    {
        Perm<N> ans;
        for (int i = 0; i < N; i++) {
            ans.ord[ord[i]] = i;
        }
        return ans;
    }

    template <int M> constexpr auto extend() const -> Perm<M>
    {
        static_assert(M >= N);
        Perm<M> ans = Perm<M>::ID();
        for (int i = 0; i < N; i++) {
            ans.ord[i] = ord[i];
        }
        return ans;
    }

    template <int M> constexpr auto shrink() const -> Perm<M>
    {
        static_assert(M <= N);
        Perm<M> ans;
        for (int i = M; i < N; i++) {
            assert(ord[i] == i);
        }
        for (int i = 0; i < M; i++) {
            ans.ord[i] = ord[i];
        }
        return ans;
    }

    // adding represent a added premutation to a fragment (see implementation)
    //
    // a layout transposition that preserve logical semantics have a
    // signature of this form:
    // template<Perm> auto transpose_layout(Frag<Perm>) -> Frag<Perm + {2, 1, 0}>
    //
    template <size_t M>
    constexpr auto operator+(const Perm<M>& p2) const -> Perm<(N > M ? N : M)>
    {
        constexpr int R = (N > M ? N : M);
        Perm<R> p1_ = this->extend<R>();
        Perm<R> p2_ = p2.template extend<R>();

        Perm<R> out;
        for (int i = 0; i < R; ++i) {
            out.ord[i] = p1_[p2_[i]];
        }
        return out;
    }

    template <typename T> constexpr auto apply(array<T, N> arr) const -> array<T, N>
    {
        array<T, N> ans;
        for (int i = 0; i < N; i++) {
            ans[ord[i]] = arr[i];
        }
        return ans;
    }
};

// nvcc type inference is failing in the transpose method with direct +
constexpr auto perm_compose(auto p1, auto p2) /* -> inferred */ { return p1 + p2; }

////////////////////////////////////////////////////////////////////////////////

// for shared mem, with 32 bank model
// how many loads does it take?
template <int limit = 1> struct SmBankCount {
    static constexpr auto check(array<int, 32> word_offsets, int pack) -> bool
    {
        array<int, 32> banks{};
        // Each lane touches 'pack' consecutive 32-bit words => 'pack' consecutive banks.
        for (int lane = 0; lane < 32; ++lane) {
            int base = word_offsets[lane];
            for (int j = 0; j < pack; ++j) {
                int bank = (base + j) % 32;
                bank = (bank + 32) % 32; // handle negative gracefully
                banks[bank] += 1;
            }
        }

        int mx = 0;
        for (int count : banks) {
            mx = std::max(mx, count);
        }

        // can be relaxed to <= if desirable
        return mx == limit * pack;
    }
};

struct Coalesced {
    static constexpr auto check(array<int, 32> word_offsets, int pack) -> bool
    {
        for (int i = 0; i < 32; i++) {
            if (word_offsets[i] != i) {
                return false;
            }
        }
        return true;
    };
};

struct MemCheckNone {
    static constexpr auto check(array<int, 32> word_offsets, int pack) -> bool
    {
        return true;
    }
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

////////////////////////////////////////////////////////////////////////////////

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

// break offsets into continuous chunks of length P
// each chunk must start with an index that is multiple of P
// negative offsets are allowed
// errors if not possible
//
template <int N, int P>
constexpr auto compute_pack(array<int, N> offsets) -> pair<
    array<int, N / P>,          // the base offsets of each chunk
    array<array<int, P>, N / P> // for each chunk: for the elements in the chunk, their
                                // corresponding index in the original offsets
>
{
    static_assert(P > 0, "P must be positive");
    static_assert(N % P == 0, "N must be divisible by P");
    constexpr int C = N / P;

    array<int, C> bases{};
    array<array<int, P>, C> idxs{};
    array<bool, N> used{};
    for (int j = 0; j < N; ++j)
        used[j] = false;

    auto modP = [](int v) constexpr -> int {
        int r = v % P;
        return (r < 0) ? (r + P) : r;
    };

    int bc = 0;

    // Discover chunks in the order their base (multiple of P) appears in offsets.
    for (int i = 0; i < N; ++i) {
        int v = offsets[i];
        if (modP(v) != 0)
            continue; // not a base

        // Ensure this base hasn't been added already
        for (int t = 0; t < bc; ++t) {
            assert(bases[t] != v && "Duplicate base encountered");
        }

        bases[bc] = v;

        // For this base, find the P consecutive elements base+k in offsets
        for (int k = 0; k < P; ++k) {
            int target = v + k;
            int found = -1;
            for (int j = 0; j < N; ++j) {
                if (!used[j] && offsets[j] == target) {
                    found = j;
                    used[j] = true;
                    break;
                }
            }
            assert(found != -1 && "Missing element for a chunk (non-contiguous)");
            idxs[bc][k] = found;
        }
        ++bc;
    }

    // Must find exactly C bases
    assert(bc == C && "Number of bases != N/P");

    // All offsets must be covered exactly once
    for (int j = 0; j < N; ++j) {
        assert(used[j] && "Offset not covered by any chunk");
    }

    // Sort chunks by base offset ascending; keep idxs rows in sync.
    for (int i = 0; i < C - 1; ++i) {
        int min_i = i;
        for (int j = i + 1; j < C; ++j) {
            if (bases[j] < bases[min_i]) {
                min_i = j;
            }
        }
        if (min_i != i) {
            int tmpb = bases[i];
            bases[i] = bases[min_i];
            bases[min_i] = tmpb;

            for (int k = 0; k < P; ++k) {
                int tmp = idxs[i][k];
                idxs[i][k] = idxs[min_i][k];
                idxs[min_i][k] = tmp;
            }
        }
    }

    return {bases, idxs};
}

template <int N, array<int, N> offsets, int PACK, typename T, typename F>
__device__ __forceinline__ auto load_offsets_batched(
    F load_fn // fn<I>() -> array<T, P>
) -> array<T, N>
{
    constexpr auto offsets_packed = compute_pack<N, PACK>(offsets);
    constexpr array<int, N / PACK> base_offsets = offsets_packed.first;
    constexpr array<array<int, PACK>, N / PACK> mapping = offsets_packed.second;

    array<T, N> ans;

    static_for<N / PACK>([&]<int I>() {
        constexpr array<int, PACK> mapping_ = mapping[I];

        array<T, PACK> tmp = load_fn.template operator()<base_offsets[I]>();
#pragma unroll
        for (int j = 0; j < PACK; j++) {
            ans[mapping_[j]] = tmp[j];
        }
    });

    return ans;
}

template <size_t N, array<int, N> offsets, int PACK, typename T, typename F>
__device__ __forceinline__ auto store_offsets_batched(
    array<T, N> data,
    F store_fn // fn<I>(array<T, P>) that stores array<T, P> at offset I
) -> void
{
    constexpr auto offsets_packed = compute_pack<N, PACK>(offsets);
    constexpr array<int, N / PACK> base_offsets = offsets_packed.first;
    constexpr array<array<int, PACK>, N / PACK> mapping = offsets_packed.second;

    static_for<N / PACK>([&]<int I>() {
        constexpr array<int, PACK> mapping_ = mapping[I];

        array<T, PACK> tmp;
#pragma unroll
        for (int j = 0; j < PACK; j++) {
            tmp[j] = data[mapping_[j]];
        }

        store_fn.template operator()<base_offsets[I]>(tmp);
    });
}

constexpr auto is_linear(array<int, 32> offsets) -> bool
{
    int base = offsets[0];
    int factor = offsets[1] - offsets[0];
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
        // assert(offsets[i] % 2 == 0);
        ans[i] = offsets[i] / 2;
    }
    return ans;
}

////////////////////////////////////////////////////////////////////////////////

struct NoMeta {};

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
template <
    DType dtype,                                      // element dtype
    size_t N,                                         // number of dimensions
    typename MetaType = NoMeta,                       //
    array<MetaType, N> AxMeta = array<MetaType, N>{}, // per logical axis metadata
    Perm<N> P = Perm<N>::ID()                         // physical -> logical mapping
>
struct Frag {
    array<u16, 1 << (N - 5)> data;

    static constexpr DType dtype_ = dtype;
    static constexpr int N_ = N;
    using MetaType_ = MetaType;
    static constexpr array<MetaType, N> AxMeta_ = AxMeta;
    static constexpr Perm<N> P_ = P;

    using Self = Frag<dtype, N, MetaType, AxMeta, P>;

    static constexpr auto perm() -> Perm<N> { return P; }

    // metadata indexed by physical axis
    static constexpr auto physical_meta() -> array<MetaType, N>
    {
        array<MetaType, N> ans;
        for (int i = 0; i < N; i++) {
            ans[i] = AxMeta[P[i]];
        }
        return ans;
    }

    // element_fn: function array<bool, N> -> u16
    //
    // [LLM] Construct a fragment by evaluating `element_fn` on every logical index
    // [LLM] element_fn: array<bool, N> -> u16. The first 5 axes are sourced from `lane`
    // [LLM] via P[0..4], and the remaining local axes from the per-thread index i via
    // P[5..N-1].
    static constexpr auto create(int lane, auto element_fn) -> Self
    {
        Self out{};

        array<bool, N> base{};
        for (int b = 0; b < 5; ++b)
            base[P[b]] = (lane >> b) & 1;

        constexpr int L = 1 << (N - 5);
        for (int i = 0; i < L; ++i) {
            auto idx = base;
            for (int k = 0; k < (N - 5); ++k)
                idx[P[5 + k]] = (i >> k) & 1;

            out.data[i] = element_fn(idx);
        }
        return out;
    }

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

    template <array<int, N> strides, int PACK, typename MEM>
    static constexpr auto mem_check_strides()
    {
        constexpr array<int, 32> offsets = get_lane_offset_all(strides);

        constexpr bool check_pack = [&]() {
            bool ok = true;
            for (auto offset : offsets) {
                if (offset % PACK != 0) {
                    ok = false;
                }
            }
            return ok;
        }();

        constexpr array<int, 32> word_offsets = [&]() {
            array<int, 32> ans;
            for (int i = 0; i < 32; i++) {
                ans[i] = offsets[i] / 2;
            }
            return ans;
        }();

        if constexpr (check_pack && MEM::check(word_offsets, div_ceil(PACK, 2))) {
            return OK();
        } else {
            return MemCheckFailed<
                array<int, 5>{
                    strides[P[0]],
                    strides[P[1]],
                    strides[P[2]],
                    strides[P[3]],
                    strides[P[4]]
                },
                MEM
            >();
        }
    }

    // get_lane_offset_impl but optentially optimized and checked with mem_check_strides
    template <array<int, N> strides>
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

    // transpose Trans by reordering logical axis, therefore doing no actual work
    template <Perm<N> Q>
    using Transposed = Frag<dtype, N, MetaType, Q.apply(AxMeta), perm_compose(Q, P)>;

    template <Perm<N> Q> constexpr auto transpose() -> Self::Transposed<Q>
    {
        return {data};
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
    template <Perm<N> Q> using PermLayout = Frag<dtype, N, MetaType, AxMeta, P + Q>;

    template <Perm<N> Q> __device__ auto transpose_layout() -> Self::PermLayout<Q>
    {
        static_assert(is_simple_trans(Q));

        Self::PermLayout<Q> out{};

        constexpr int L = 1 << (N - 5);
        if constexpr (L == 1) {
            out.data[0] = data[0];
            return out;
        }

        // Invert Trans over full N (cheap); we only use the local part.
        constexpr auto inv = Q.inv();

        // For each old local index, compute its destination index by permuting
        // bit-positions.
#pragma unroll
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
        int PACK,
        typename MEM = SmBankCount<1>,
        typename = CHECK_V<mem_check_strides<strides, PACK, MEM>()>,
        typename F
    >
    __device__ static auto load_with(const u16* ptr, int lane, F load_fn) -> Self
    {
        ptr += get_lane_offset<strides>(lane);
        ptr = assert_aligned(ptr, sizeof(u16) * PACK);

        constexpr auto offsets = get_local_offsets(strides);

        auto ans = load_offsets_batched<offsets.size(), offsets, PACK, u16>(
            [&]<int I>() -> array<u16, PACK> {
                static_assert(I % PACK == 0);
                return load_fn(ptr + I);
            }
        );
        return {ans};
    }

    template <
        array<int, N> strides = default_strides<N>(), // logical strides
        typename MEM = SmBankCount<1>,
        int PACK = 2,
        typename = CHECK_V<mem_check_strides<strides, PACK, MEM>()>
    >
    __device__ static auto load(const u16* ptr, int lane) -> Self
    {
        return load_with<strides, PACK, MEM>(
            ptr,
            lane,
            [](const u16* ptr) -> array<u16, PACK> {
                array<u16, PACK> piece;

                if constexpr (PACK == 1) {
                    piece[0] = *ptr;
                } else {
                    // pack to u32
                    static_assert(PACK % 2 == 0);
                    const u32* ptr_u32 = reinterpret_cast<const u32*>(ptr);
                    u32 piece_u32[PACK / 2];

                    if constexpr (PACK == 2) {
                        piece_u32[0] = *ptr_u32;
                    } else if constexpr (PACK == 8) {
                        cp_16(piece_u32, ptr_u32);
                    } else {
                        typename Dummy2<PACK>::x x;
                    }

#pragma unroll
                    for (int i = 0; i < PACK / 2; i++) {
                        piece[2 * i] = lo16(piece_u32[i]);
                        piece[2 * i + 1] = hi16(piece_u32[i]);
                    }
                }

                return piece;
            }
        );
    }

    template <
        array<int, N> strides = default_strides<N>(), // logical strides
        int PACK,
        typename MEM = SmBankCount<1>,
        typename = CHECK_V<mem_check_strides<strides, PACK, MEM>()>,
        typename F
    >
    __device__ auto store_with(u16* ptr, int lane, F store_fn) const -> void
    {
        ptr += get_lane_offset<strides>(lane);
        ptr = assert_aligned(ptr, sizeof(u16) * PACK);

        constexpr auto offsets = get_local_offsets(strides);

        store_offsets_batched<offsets.size(), offsets, PACK>(
            data,
            [&]<int I>(array<u16, PACK> piece) {
                static_assert(I % PACK == 0);
                store_fn(ptr + I, piece);
            }
        );
    }

    template <
        array<int, N> strides = default_strides<N>(), // logical strides
        typename MEM = SmBankCount<1>,
        int PACK = 2,
        typename = CHECK_V<mem_check_strides<strides, PACK, MEM>()>
    >
    __device__ auto store(u16* ptr, int lane) const -> void
    {
        store_with<strides, PACK, MEM>(ptr, lane, [](u16* ptr, array<u16, PACK> piece) {
            if constexpr (PACK == 1) {
                *ptr = piece[0];
            } else {
                // pack to u32
                static_assert(PACK % 2 == 0);
                u32* ptr_u32 = reinterpret_cast<u32*>(ptr);
                u32 piece_u32[PACK / 2];
                for (int i = 0; i < PACK / 2; i++) {
                    piece_u32[i] = pack_b16x2(piece[2 * i], piece[2 * i + 1]);
                }

                if constexpr (PACK == 2) {
                    *ptr_u32 = piece_u32[0];
                } else if constexpr (PACK == 8) {
                    cp_16(ptr_u32, piece_u32);
                } else {
                    typename Dummy2<PACK>::x x;
                }
            }
        });
    }

    // <logical behavior>
    // call a function with a different perm spec by
    //   transposing, apply function, then transposing back
    //
    // args:
    //   fn: function(Frag<Perm<M> ArgP>) -> Frag<Perm<M> ...>
    //
    // returns:
    //   T_inv(fn(T(this)))
    //
    //  T is logical transpose that does no physical work:
    //   T[logical axis mapping]:= (ArgP[0] -> P[0], ArgP[1] -> P[1], ...)
    //
    // if N is larger M, this function extends fn:
    // the rest of the dimensions are batch dims.
    //
    // <physical behavior>
    //
    // fn is a function with the wrong axis ordering.
    // we apply it to argument arr anyways pretending it is right,
    // and give a reasonable output axis labeling.
    //
    template <int M, Perm<M> ArgP = Perm<M>::ID()>
    __host__ __device__ __forceinline__ constexpr auto apply_transposed(
        auto fn
    ) /* -> inferred */
    {
        static_assert(M <= N);
        static_assert(M >= 5, "apply_transposed requires M >= 5 (at least 5 lane dims).");

        // T maps our current layout P to ArgP (extended to N) without moving data.
        // We want (P + T)[i] == ArgP[i] for i < M, and identity for i >= M.
        constexpr Perm<N> T = ArgP.template extend<N>() + P.inv();

        // Logical transpose into the ArgP basis (no data movement).
        auto self_trans = this->template transpose<T>();
        using ArgT_ = decltype(self_trans);

        // Ensure we matched the requested ArgP on the first M logical axes.
        []() consteval {
            for (int i = 0; i < M; i++) {
                assert(ArgT_::P_[i] == ArgP[i]);
            }
        }();

        // Argument type expected by fn (first M axes only).
        using ArgT = Frag<dtype, M, MetaType, slice_array<M>(ArgT_::AxMeta_), ArgP>;

        // Result type of fn; must also be M-dimensional.
        using RetT = decltype(fn(std::declval<ArgT>()));
        static_assert(RetT::N_ == M);

        // Pre-inverse-transpose output meta: first M axes from RetT, batch axes from
        // ArgT_.
        constexpr auto OutAxMetaPre = []() {
            array<MetaType, N> out{};
            for (int i = 0; i < M; ++i)
                out[i] = RetT::AxMeta_[i];
            for (int i = M; i < N; ++i)
                out[i] = ArgT_::AxMeta_[i];
            return out;
        }();

        // Pre-inverse-transpose output perm: RetT perm extended with identity on batch
        // axes.
        constexpr Perm<N> OutPermPre = RetT::P_.template extend<N>();

        using OutPre = Frag<dtype, N, MetaType, OutAxMetaPre, OutPermPre>;
        OutPre out_pre{};

        // Local element counts.
        constexpr int L_small = 1 << (M - 5); // per-thread elements for M dims
        constexpr int B_bits =
            N - M; // number of batch local bits (all local, since M >= 5)
        constexpr int B_count = 1 << B_bits;

        // For each batch-slice, extract M-dim arg, call fn, and insert back.
#pragma unroll
        for (int b = 0; b < B_count; ++b) {
            ArgT arg_small{};
            // Extract slice from self_trans into arg_small.
#pragma unroll
            for (int i = 0; i < L_small; ++i) {
                int idx_big = i | (b << (M - 5));
                arg_small.data[i] = self_trans.data[idx_big];
            }

            // Apply fn on the M-dim slice.
            auto ret_small = fn(arg_small);

            // Stitch back into out_pre at the same batch offset.
#pragma unroll
            for (int i = 0; i < L_small; ++i) {
                int idx_big = i | (b << (M - 5));
                out_pre.data[idx_big] = ret_small.data[i];
            }
        }

        // Finally, invert the initial logical transpose to return to the original basis.
        constexpr Perm<N> T_inv = T.inv();
        return out_pre.template transpose<T_inv>();
    }
};

constexpr Perm<8> PermA = {1, 2, 4, 5, 6, /**/ 0, 7, 3};
constexpr Perm<7> PermB = {1, 2, 4, 5, 6, /**/ 0, 3};
constexpr Perm<7> PermC = {5, 6, 0, 1, 2, /**/ 4, 3};

// <logical behavior>
// Compute a generalized dot product:
// A[k1, k2, k3, k4, m1, m2, m3, m4]
// B[k1, k2, k3, k4, n1, n2, n3]
// ->
// C[m1, m2, m3, m4, n1, n2, n3]
// (so the first 4 logical axis of A and B disappears by dot product with each other)
// <logical description>
// These tensors must be layed out in registers as required in the type signature.
//
// A single tensor core call is used. No other computation or memory access is used.
//
// <physical behavior>
// Applies a tensor core mma,
// labeling axis according to nvidia swizzle and therefore
// delegating swizzling to the caller
template <DType dtype, typename MetaType, array<MetaType, 8> MA, array<MetaType, 7> MB>
__device__ __forceinline__ auto mma_m16_n8_k16(
    Frag<dtype, 8, MetaType, MA, PermA> A,
    Frag<dtype, 7, MetaType, MB, PermB> B
)
    -> Frag<
        dtype,
        7,
        MetaType,
        array<MetaType, 7>{MA[4], MA[5], MA[6], MA[7], MB[4], MB[5], MB[6]},
        PermC
    >
{
    // Pack A (8 scalars -> 4x b16x2), B (4 scalars -> 2x b16x2).
    array<u32, 4> a{
        pack_b16x2(A.data[0], A.data[1]),
        pack_b16x2(A.data[2], A.data[3]),
        pack_b16x2(A.data[4], A.data[5]),
        pack_b16x2(A.data[6], A.data[7]),
    };
    array<u32, 2> b{
        pack_b16x2(B.data[0], B.data[1]),
        pack_b16x2(B.data[2], B.data[3]),
    };

    constexpr u32 z = 0;
    u32 c0, c1;

    if constexpr (dtype == DType::Half) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                     "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
                     : "=r"(c0), "=r"(c1)
                     : "r"(a[0]),
                       "r"(a[1]),
                       "r"(a[2]),
                       "r"(a[3]),
                       "r"(b[0]),
                       "r"(b[1]),
                       "r"(z),
                       "r"(z));
    } else {
        // Assume bf16; accumulate in fp32 and pack to bf16x2.
        u32 t0, t1, t2, t3;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
            : "r"(a[0]),
              "r"(a[1]),
              "r"(a[2]),
              "r"(a[3]),
              "r"(b[0]),
              "r"(b[1]),
              "r"(z),
              "r"(z),
              "r"(z),
              "r"(z)
        );
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(c0) : "r"(t1), "r"(t0));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(c1) : "r"(t3), "r"(t2));
    }

    // Unpack C per-lane in hardware order: i={0,1,2,3} => {lo(c0), hi(c0), lo(c1),
    // hi(c1)}.
    return {{lo16(c0), hi16(c0), lo16(c1), hi16(c1)}};
}
