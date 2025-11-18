#include "defs.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

using u16 = uint16_t;
using u32 = uint32_t;
using std::array;
using std::pair;
using DType = torch::ScalarType;

__host__ __device__ void assume(bool cond)
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

// ----------------------------- Permutations ---------------------------------
//
// Perm represents a permutation of N logical axes. Composition is defined so that
// (P + Q)[i] = P[Q[i]] (i.e., Q is applied first, then P). We use this to track
// how data is distributed across lanes and registers without physically moving it.
//
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

    // Compose permutations of possibly different sizes.
    // Semantics: (this ∘ p2) over max(N, M), padding the smaller one with identity.
    // For i >= size(p2), p2 acts as identity; for k >= size(this), this acts as identity.
    // Complexity: O(max(N, M)).
    template <int M>
    constexpr auto operator+(const Perm<M>& p2) const -> Perm<(N > M ? N : M)>
    {
        constexpr int R = (N > M ? N : M);
        Perm<R> out{};
        for (int i = 0; i < R; ++i) {
            const int a = (i < M) ? p2[i] : i; // g(i): p2 or identity
            out.ord[i] = (a < N) ? ord[a] : a; // f(a): this or identity
        }
        return out;
    }
};

// ------------------- Small helpers (packing and bits) -----------------------

__device__ __forceinline__ u32 pack_b16x2(u16 lo, u16 hi)
{
    return static_cast<u32>(lo) | (static_cast<u32>(hi) << 16);
}
__device__ __forceinline__ u16 lo16(u32 x) { return static_cast<u16>(x & 0xFFFFu); }
__device__ __forceinline__ u16 hi16(u32 x) { return static_cast<u16>(x >> 16); }

// Build bit patterns for ±1/4 in f16/bf16. This is the scale factor for a 4-axis
// Hadamard: (1/sqrt(2))^4 = 0.25. Using constants here avoids extra FP ops.
//
// +1/4
template <DType dtype> __device__ __forceinline__ u16 quarter_pos_bits()
{
    if constexpr (dtype == DType::Half)
        return static_cast<u16>(0x3400); // 0.25 in f16
    else
        return static_cast<u16>(0x3E80); // 0.25 in bf16
}
// -1/4
template <DType dtype> __device__ __forceinline__ u16 quarter_neg_bits()
{
    return static_cast<u16>(quarter_pos_bits<dtype>() | 0x8000u);
}

// Default linear strides for a 7D 2x...x2 tensor: stride[ax] = 1 << ax.
// With this convention bit ax toggles every stride[ax].
//
template <int N> constexpr auto default_strides() -> array<int, N>
{
    array<int, N> ans{};
    for (int i = 0; i < N; i++)
        ans[i] = 1 << i;
    return ans;
}

// ------------------------- Fragment view (Frag) -----------------------------
//
// Frag<N,P,dtype> is a tiny logical view over the 128-element (N=7) warp tile,
// distributed across lanes as imposed by the MMA instruction. Each lane owns
// 2^(N-5) scalars (for N=7, that's 4 scalars).
//
// You can think of it as:
// - get_lane_offset(): which base element this lane starts from (based on the
//   lower 5 logical axes of P).
// - get_local_offsets(): the 2^(N-5) “local” increments for the remaining axes.
//
// IMPORTANT: Both lane and local offsets are computed with the SAME permutation P.
// This keeps the logical axes coherent even when the hardware shuffles bits.
// A prior bug ignored P for local offsets; do not revert that.
//
// Usage:
// - load(ptr, lane): read this lane’s 4 scalars from memory into registers
//   (no reordering; just a fixed pattern).
// - store(ptr, lane): write them back in the same logical coordinate system.
//
// Frag has value semantics. It is used to represent in-register data. it does not make
// sense to create a Frag on shared or global memory.
template <DType dtype, int N, Perm<N> P = Perm<N>::ID()> struct Frag {
    array<u16, 1 << (N - 5)> data;

    static constexpr auto perm() -> Perm<N> { return P; }

    // Sum strides of the 5 lane-controlled bits present in this lane ID.
    __device__ static auto get_lane_offset(array<int, N> strides, int lane) -> int
    {
        int ans = 0;
#pragma unroll
        for (int i = 0; i < 5; i++)
            if (lane & (1 << i))
                ans += strides[P[i]];
        return ans;
    }

    // transpose layout without changing logical array content
    // currently only among local dimension is supported
    template <Perm<N> Trans>
    __device__ auto transpose_layout() -> Frag<dtype, N, P + Trans>
    {
        for (int i = 0; i < 5; i++) {
            assert(Trans[i] == i);
        }

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
                    const int k_new = inv[5 + k] - 5; // where old bit k lands
                    dst |= (1 << k_new);
                }
            }
            out.data[dst] = data[src];
        }
        return out;
    }

    // Offsets for the (N-5) “local” axes (the ones not mapped to %laneid bits).
    // NOTE: We must index strides using P[j+5], not raw j+5. This ensures local
    // axes track the fragment’s permutation consistently with the lane axes.
    static constexpr auto get_local_offsets(array<int, N> strides)
        -> array<int, 1 << (N - 5)>
    {
        array<int, 1 << (N - 5)> ans{};
        for (int i = 0; i < int(ans.size()); i++) {
            int ofs = 0;
            for (int j = 0; j < N - 5; j++)
                if (i & (1 << j))
                    ofs += strides[P[j + 5]];
            ans[i] = ofs;
        }
        return ans;
    }

    // Load 2^(N-5) scalars for this lane.
    template <array<int, N> strides = default_strides<N>()>
    __device__ static auto load(const u16* ptr, int lane) -> Frag<dtype, N, P>
    {
        ptr += get_lane_offset(strides, lane);
        constexpr auto offsets = get_local_offsets(strides);

        array<u16, 1 << (N - 5)> data{};
#pragma unroll
        for (int i = 0; i < int(data.size()); i++)
            data[i] = ptr[offsets[i]];
        return {data};
    }

    // Store 2^(N-5) scalars for this lane.
    template <array<int, N> strides = default_strides<N>()>
    __device__ auto store(u16* ptr, int lane) const -> void
    {
        ptr += get_lane_offset(strides, lane);
        constexpr auto offsets = get_local_offsets(strides);
#pragma unroll
        for (int i = 0; i < int(data.size()); i++)
            ptr[offsets[i]] = data[i];
    }
};

// ------------------------ Hadamard matrix (A fragment) ----------------------
//
// We build the 16x16 Hadamard “A” matrix once as booleans and then shard it into
// per-lane bitmasks following the official NVIDIA mapping for m16n8k16.row.col
// (see “9.7.14.5.8. Matrix Fragments for mma.m16n8k16 with floating point type”).
//
// For matrix A:
//   groupID           = %laneid >> 2
//   threadID_in_group = %laneid & 0x3
//
//   row = groupID                  for i in {0,1,4,5}
//         groupID + 8              for i in {2,3,6,7}
//
//   col = (threadID_in_group * 2) + (i & 1)     for i in {0,1,2,3}
//         (threadID_in_group * 2) + (i & 1) + 8 for i in {4,5,6,7}
//
// The code below encodes exactly this mapping, producing eight 32-bit masks,
// one mask per A-operand register ai (i=0..7). Bit L in mask ai is 1 if, for
// lane L, ai should carry +1/4; otherwise -1/4.
//
struct HadMat {
    array<array<bool, 16>, 16> data;

    // Build the standard 16x16 Hadamard (+1/-1) by parity of popcount(i & j).
    constexpr static auto create_had() -> HadMat
    {
        HadMat ans{};
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                ans.data[i][j] = (__builtin_popcount(i & j) % 2) == 0;
        return ans;
    }

    // Shard the matrix into warp-lane masks using a callback that provides
    // the (row, col) per ai register index i=0..7 for each lane.
    constexpr auto shard(auto cb) const -> array<u32, 8>
    {
        array<u32, 8> ans{};
        for (int lane = 0; lane < 32; lane++) {
            auto pos = cb(lane);
            for (int j = 0; j < 8; j++) {
                auto [x, y] = pos[j];
                if (data[x][y])
                    ans[j] |= (1u << lane);
            }
        }
        return ans;
    }
};

// ------------------------- rotate_4 (core op) -------------------------------
//
// rotate_4 logically computes a 4-axis Hadamard transform on axes {0,1,5,6}
// of a 7D 2x...x2 tile, scaled by 0.25. It returns data expressed in the SAME
// logical coordinate system as the input Frag, so callers never see any axis
// permutation side-effects.
//
// Internals:
// - We run one MMA per warp: C = A @ B, where A is 16x16 Hadamard scaled by 1/4,
//   B comes from arr, and C is the result we pack back into a Frag.
// - A’s per-lane row/col coordinates follow the NVIDIA documentation mapping;
//   we encode them as bitmasks (masks[]) that decide whether each ai register
//   holds +0.25 or -0.25 for this lane.
// - B’s register layout is the standard one for the “row.col” variant and directly
//   consumes arr.data[] in the prescribed order.
// - Hardware returns C in a per-lane layout that effectively permutes logical axes.
//   We experimentally measured this and compensate via the return Frag’s permutation,
//   so the caller sees a clean logical transform.
//
// Caution:
// - Passing a different HadMat than HadMat::create_had() is not tested. The mask
//   layout code is general but our tests only validate the standard Hadamard.
//

template <DType dtype, HadMat H = HadMat::create_had()>
__device__ __forceinline__ auto rotate_4(const Frag<dtype, 7> arr, int lane)
    // IMPORTANT: The hardware introduces a fixed permutation for the C fragment.
    // We compensate for it here so that rotate_4() behaves as a pure Hadamard on
    // axes {0,1,5,6} in logical coordinates. The measured compensation is:
    //   Comp = {3,4,5,0,1,2,6}
    // Hence the result fragment is annotated with P + Comp.
    -> Frag<dtype, 7, Perm<7>{3, 4, 5, 0, 1, 2, 6}>
{
    // Build the A fragment signs (+/- 0.25) once per compile as lane-masks.
    // This lambda encodes the official A-fragment mapping (see big comment above).
    constexpr array<u32, 8> masks = H.shard([](int lane) -> array<pair<int, int>, 8> {
        const int gid = lane >> 2; // groupID
        const int tid = lane & 3;  // threadID_in_group
        array<pair<int, int>, 8> pos{};
        for (int i = 0; i < 8; ++i) {
            // Row selector: i in {0,1,4,5} => gid; i in {2,3,6,7} => gid+8
            const bool m_hi_bit = (i >= 2 && i < 4) || (i >= 6);
            const int m = gid + (m_hi_bit ? 8 : 0);
            // Column selector: i<4 => base; i>=4 => base+8
            const int k_base = (tid * 2) + (i & 1);
            const int k = k_base + (i >= 4 ? 8 : 0);

            // pos[i] = {m, k}; // nvcc is happy but clangd is not for some reason
            pos[i].first = m;
            pos[i].second = k;
        }
        return pos;
    });

    // Turn masks into per-lane ai register values (±0.25 in the chosen dtype).
    const u16 qpos = quarter_pos_bits<dtype>();
    const u16 qneg = quarter_neg_bits<dtype>();
    array<u32, 4> a{};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const u16 lo = (masks[2 * i + 0] & (1u << lane)) ? qpos : qneg;
        const u16 hi = (masks[2 * i + 1] & (1u << lane)) ? qpos : qneg;
        a[i] = pack_b16x2(lo, hi);
    }

    // Pack B as specified by m16n8k16.row.col; arr.data[] are in the correct order.
    array<u32, 2> b{};
    b[0] = pack_b16x2(arr.data[0], arr.data[1]);
    b[1] = pack_b16x2(arr.data[2], arr.data[3]);

    // Run the MMA. For bf16 we accumulate in fp32 and convert back to bf16x2 at the end.
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
        // Pack two fp32 accumulators into a single bf16x2 register each.
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(c0) : "r"(t1), "r"(t0));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(c1) : "r"(t3), "r"(t2));
    }

    // Return four scalars in lane order (c00,c01,c10,c11). The Frag’s permutation
    // annotation ensures store() writes them back at the correct logical indices.
    return {{lo16(c0), hi16(c0), lo16(c1), hi16(c1)}};
}

//////////

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

// arg:
//  map_fn: (Frag, lane) -> Frag function. This only need to support a standard layout
//   for Frag.
//  arr: Frag with axis spec P. This operation will be P dependent.
//
// This function operates on a submatrix given by axis P[0], P[1], .. , P[N-1]
// (which are the logical axis corresponding to the first N pysical axis).
//
// the map_fn function will be run on each slice of the submatrix
// P[0] will be passed as the 0th logical axis of map_fn
//
// Returns of map_fn will be aggregated and returned
template <int N, int M, Perm<M> P, DType dtype>
__host__ __device__ __forceinline__ auto apply_on_local_prefix(
    const Frag<dtype, M, P> arr,
    auto map_fn,
    int lane
)
{
    static_assert(N <= M);
    constexpr Perm<N> out_perm =
        decltype(map_fn(std::declval<Frag<dtype, N, Perm<N>::ID()>>(), 0))::perm();
    constexpr int D = 1 << (M - N); // batch dimensions
    constexpr int S = 1 << (N - 5); // #count of u16
    static_assert(arr.data.size() == D * S);

    using OutType = Frag<dtype, M, P + out_perm>;
    BlackBox<OutType> ans;

    for (int i = 0; i < D; i++) {
        Frag<dtype, N, Perm<N>::ID()> sub_mat;
        for (int j = 0; j < S; j++) {
            sub_mat.data[j] = arr.data[S * i + j];
        }

        Frag<dtype, N, out_perm> res = map_fn(sub_mat, lane);

        for (int j = 0; j < S; j++) {
            ans.data[S * i + j] = res.data[j];
        }
    }

    return ans;
}

// constexpr DType dtype = DType::BFloat16;

// rotate all 8 axis
// rotate_4 transposition happen to square to the identity, so no logical transpose here
template <DType dtype>
__device__ auto hada_local_8_8(const Frag<dtype, 8> frag, int lane) -> Frag<dtype, 8>
{
    // /**/ marked axis are rotated
    Frag<dtype, 8, Perm<8>{3, 4, 5 /**/, 0 /**/, 1 /**/, 2, 6 /**/, 7}> tmp1 =
        // roates position [0, 1, 5, 6]
        apply_on_local_prefix<7>(frag, rotate_4<dtype>, lane);

    Frag<dtype, 8, Perm<8>{3, 4, 5 /**/, 0 /**/, 1 /**/, 2, 7, 6 /**/}> tmp2 =
        tmp1.template transpose_layout<Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}>();

    // all is rotated now
    Frag<dtype, 8, Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}> tmp3 =
        apply_on_local_prefix<7>(tmp2, rotate_4<dtype>, lane);

    // not mandatory but its free to make it identity
    return tmp3.template transpose_layout<Perm<8>{0, 1, 2, 3, 4, 5, 7, 6}>();
}

/////////////////////////////////////////////////

template <DType dtype, int N> struct RowHandler {};

template <DType dtype> struct RowHandler<dtype, 8> {
    __device__ auto handle_row(const u16* in, u16* out) -> void
    {
        int lane = threadIdx.x;
        assert_(0 <= lane);
        assert_(lane < 32);

        auto in_reg = Frag<
            dtype,
            8,
            Perm<8>{
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
            }
        >::load(in, lane);

        auto out_reg = apply_on_local_prefix<8>(in_reg, hada_local_8_8<dtype>, lane);

        __syncwarp();
        out_reg.store(out, lane);
    }
};

template <DType dtype, int N>
__global__ auto hadamard_transform_ker(const u16* a, u16* out, int num_rows) -> void
{
    RowHandler<dtype, N> handler{};

    int i = blockIdx.x;
    const u16* a_i = a + i * (1 << N);
    u16* out_i = out + i * (1 << N);

    handler.handle_row(a_i, out_i);
}

template <torch::ScalarType dtype>
auto run_fht(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
) -> void
{
    TORCH_CHECK(had_size == 256, "This implementation only supports Hadamard size 256");

    const uint32_t num_rows = numel / 256;

    hadamard_transform_ker<dtype, 8><<<num_rows, 32, 0, stream>>>(
        static_cast<const u16*>(a_mat_ptr),
        static_cast<u16*>(out_ptr),
        static_cast<int>(num_rows)
    );

    // TORCH_CHECK((numel % 256) == 0, "numel must be divisible by 256");

    // const uint32_t num_rows = numel / 256;
    // dim3 grid(num_rows), block(32);
    // constexpr size_t shmem = 128 * sizeof(u32); // 512B

    // hadamard_transform_256_kernel<dtype><<<grid, block, shmem, stream>>>(
    //     static_cast<const u16*>(a_mat_ptr),
    //     static_cast<u16*>(out_ptr),
    //     static_cast<int>(num_rows)
    // );
    // C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Explicit instantiations (fp16, bf16)
template auto run_fht<torch::ScalarType::Half>(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
) -> void;
template auto run_fht<torch::ScalarType::BFloat16>(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
) -> void;

// /////////////////////////////////////////////////
// // AI generated test/debug code for test_rotate4
// // use with caution

// // -------------------- Dtype and reference helpers ---------------------------
// //
// // Utility conversions for building a reference Hadamard on the device.
// //
// template <DType dtype> __device__ __forceinline__ float bits_to_float(u16 bits)
// {
//     if constexpr (dtype == DType::Half)
//         return __half2float(__ushort_as_half(bits));
//     else
//         return __bfloat162float(__ushort_as_bfloat16(bits));
// }

// template <DType dtype> __device__ __forceinline__ u16 float_to_bits(float x)
// {
//     if constexpr (dtype == DType::Half)
//         return __half_as_ushort(__float2half_rn(x));
//     else
//         return __bfloat16_as_ushort(__float2bfloat16_rn(x));
// }

// // In-place 1D Hadamard along axis ax for a 7D 2x...x2 tile (N must be 7).
// // This is the standard butterfly with normalization (1/sqrt(2)).
// //
// template <DType dtype>
// __device__ __forceinline__ void hadamard_axis_inplace(u16* buf, int ax)
// {
//     const int stride = 1 << ax;
//     const float s = rsqrtf(2.0f);
// #pragma unroll
//     for (int i = 0; i < 128; ++i) {
//         if (!(i & stride)) {
//             const int i0 = i;
//             const int i1 = i + stride;
//             const float x = bits_to_float<dtype>(buf[i0]);
//             const float y = bits_to_float<dtype>(buf[i1]);
//             buf[i0] = float_to_bits<dtype>((x + y) * s);
//             buf[i1] = float_to_bits<dtype>((x - y) * s);
//         }
//     }
// }

// // ------------------- Kernels (test scaffolding) -----------------------------
// //
// // identity_input_kernel: write a one-hot column at index trace_idx.
// // test_rotate_4: compare rotate_4 against a straightforward reference
// //                implementation that runs the 4 one-axis Hadamards.
// // kernel_rotate_4: limit execution to the first warp of a block.
// //
// template <DType dtype> __global__ void identity_input_kernel(u16* in128, int trace_idx)
// {
//     const int i = threadIdx.x;
//     if (i < 128)
//         in128[i] = float_to_bits<dtype>((i == trace_idx) ? 1.0f : 0.0f);
// }

// template <Perm<7> P, DType dtype>
// __device__ void test_rotate_4(const u16* in128, u16* out1, u16* out2)
// {
//     const int lane = threadIdx.x & 31;

//     // Build reference on a single lane to avoid redundant work.
//     if (lane == 0) {
//         for (int i = 0; i < 128; ++i)
//             out1[i] = in128[i];
//         for (int ax_local : {0, 1, 5, 6})
//             hadamard_axis_inplace<dtype>(out1, P[ax_local]);
//     }
//     __syncwarp();

//     // Call the Tensor Core path.
//     const auto in = Frag<dtype, 7, P>::load(in128, lane);
//     const auto rotated = rotate_4<dtype>(in, lane);
//     rotated.store(out2, lane);

//     __syncwarp();
// }

// template <DType dtype>
// __global__ void kernel_rotate_4(const u16* in128, u16* out1, u16* out2)
// {
//     // Only the first warp participates.
//     if ((threadIdx.x & ~31) != 0)
//         return;
//     test_rotate_4<Perm<7>::ID(), dtype>(in128, out1, out2);
// }

// // ------------------- Host harness (self-test) -------------------------------
// //
// // The functions below construct a column of the 7D Hadamard applied
// // on axes {0,1,5,6} in host float, compare it with device results,
// // and print the discovered mapping (should be identity on success).
// //
// template <DType dtype> auto host_bits_to_float(u16 bits) -> float
// {
//     if constexpr (dtype == DType::Half) {
//         at::Half h;
//         memcpy(&h, &bits, sizeof(u16));
//         return float(h);
//     } else {
//         at::BFloat16 b;
//         memcpy(&b, &bits, sizeof(u16));
//         return float(b);
//     }
// }

// // Reference column for a one-hot input at col_idx, after applying the 4-axis
// // Hadamard on axes {0,1,5,6}. Batch axes {2,3,4} are identity.
// auto get_reference_hadamard_col(int col_idx) -> std::vector<float>
// {
//     std::vector<float> col(128);
//     float scale = 0.25f; // (1/sqrt(2))^4 from 4 axes
//     for (int row_idx = 0; row_idx < 128; ++row_idx) {
//         int had_axes_mask = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 6);
//         int val = (row_idx & had_axes_mask) & (col_idx & had_axes_mask);
//         col[row_idx] = (__builtin_popcount(val) % 2 == 0) ? scale : -scale;

//         // Apply identity for batch axes
//         int batch_axes_mask = (1 << 2) | (1 << 3) | (1 << 4);
//         if ((row_idx & batch_axes_mask) != (col_idx & batch_axes_mask))
//             col[row_idx] = 0.0f;
//     }
//     return col;
// }

// // Return the ref column index that best matches mma_output, or -1 if none is close.
// template <DType dtype> static int find_best_match(const array<u16, 128>& mma_output)
// {
//     int best_match_idx = -1;
//     float min_err = 1e9f;

//     for (int ref_idx = 0; ref_idx < 128; ++ref_idx) {
//         auto ref_col = get_reference_hadamard_col(ref_idx);
//         float current_err = 0.0f;
//         for (int i = 0; i < 128; ++i) {
//             float diff = host_bits_to_float<dtype>(mma_output[i]) - ref_col[i];
//             current_err += diff * diff;
//         }
//         if (current_err < min_err) {
//             min_err = current_err;
//             best_match_idx = ref_idx;
//         }
//     }
//     // Heuristic to declare "no match"
//     if (min_err > 1.0f)
//         return -1;
//     return best_match_idx;
// }

// template <DType dtype> static bool run_rotate4_test()
// {
//     auto cucheck = [](cudaError_t e, const char* what) {
//         if (e != cudaSuccess) {
//             std::cerr << what << " failed: " << cudaGetErrorString(e) << std::endl;
//             std::terminate();
//         }
//     };

//     u16 *d_in, *d_out1, *d_out2;
//     cucheck(cudaMalloc(&d_in, 128 * sizeof(u16)), "malloc in");
//     cucheck(cudaMalloc(&d_out1, 128 * sizeof(u16)), "malloc out1");
//     cucheck(cudaMalloc(&d_out2, 128 * sizeof(u16)), "malloc out2");

//     array<u16, 128> h_out2{};
//     std::vector<int> permutation_map(128);
//     int mismatches = 0;

//     const char* name = (dtype == DType::Half) ? "f16" : "bf16";
//     printf("--- Discovering permutation map for %s ---\n", name);

//     for (int trace_idx = 0; trace_idx < 128; ++trace_idx) {
//         identity_input_kernel<dtype><<<1, 128>>>(d_in, trace_idx);
//         kernel_rotate_4<dtype><<<1, 32>>>(d_in, d_out1, d_out2);
//         cucheck(cudaDeviceSynchronize(), "kernel sync");

//         cucheck(
//             cudaMemcpy(h_out2.data(), d_out2, 128 * sizeof(u16),
//             cudaMemcpyDeviceToHost), "D->H out2"
//         );

//         int actual_output_idx = find_best_match<dtype>(h_out2);
//         permutation_map[trace_idx] = actual_output_idx;
//         if (trace_idx != actual_output_idx)
//             mismatches++;
//     }

//     printf("Discovered mapping (input_idx -> actual_output_idx):\n");
//     for (int i = 0; i < 128; i += 8) {
//         for (int j = 0; j < 8; ++j)
//             printf("  %3d->%-3d ", i + j, permutation_map[i + j]);
//         printf("\n");
//     }

//     if (mismatches == 0) {
//         printf(
//             "[OK] rotate_4 test (%s): The hardcoded permutation appears correct.\n",
//             name
//         );
//     } else {
//         printf(
//             "[FAIL] rotate_4 test (%s): Found %d mismatches in permutation map.\n",
//             name,
//             mismatches
//         );
//         printf(
//             "The map above IS the permutation the hardware performs. Update the Frag "
//             "return type in rotate_4 to match it.\n"
//         );
//     }

//     cucheck(cudaFree(d_in), "free in");
//     cucheck(cudaFree(d_out1), "free out1");
//     cucheck(cudaFree(d_out2), "free out2");

//     return mismatches == 0;
// }

// void test_rotate4()
// {
//     run_rotate4_test<DType::Half>();
//     run_rotate4_test<DType::BFloat16>();
// }

/////////////////////////////////////////////////
