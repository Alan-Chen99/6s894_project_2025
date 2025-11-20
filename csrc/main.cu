// comment out to enable assert
#define NDEBUG

#include "defs.cuh"
#include "frag.cuh"
#include "utils.cuh"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

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
    static constexpr auto create_had() -> HadMat
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

    // TODO:
    // Dummy2<masks>::x x;
    // masks apparently only have 3 distinct values, why?

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

// load M * 256 element from global in to shared out
// every 256 consecutive element will be hadamard transformed
template <
    DType dtype,
    int N,
    int P, // # of cp.async pipeline at once
    int S
    // array<int, N> sm_strides // layout in shared mem
>
__device__ auto load_rot_8(const u16* in, u16* sm, int lane) -> void
{
    static_for<N + P>([&]<int I>() {
        if constexpr (I < N) {
            cp_async16(
                sm + I * S + lane * 8,  // shared: stride S
                in + I * 256 + lane * 8 // gobal: stride 256
            );
            async_commit_group();
        }

        if constexpr (P <= I) {
            constexpr int idx = I - P;
            constexpr int ongoing = std::min(I + 1, N);

            async_wait_pending<ongoing - idx - 1>();
            __syncwarp();

            u16* data = sm + idx * S;

            auto in_reg = Frag<
                dtype,
                8,
                Perm<8>{1, 2, 3, 4, 5, 0 /*u16->u32 packing axis*/, 6, 7}
            >::load(data, lane);
            auto out_reg = apply_on_local_prefix<8>(in_reg, hada_local_8_8<dtype>, lane);
            out_reg.store(data, lane);
        }
    });
}

/////////////////////////////////////////////////

template <DType dtype, int N> struct RowHandler;

template <DType dtype> struct RowHandler<dtype, 8> {
    static constexpr int rows_per_block = 4;

    __device__ auto handle_row(const u16* in, u16* out) -> void
    {
        int lane = threadIdx.x;
        assert_((0 <= lane) && (lane < 32));

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

        auto in_reg = Frag<dtype, 8, perm>::template load<strides, Coalesced>(in, lane);

        auto out_reg = apply_on_local_prefix<8>(in_reg, hada_local_8_8<dtype>, lane);

        out_reg.template store<strides, Coalesced>(out, lane);
    }
};

template <DType dtype> struct RowHandler<dtype, 12> {
    static constexpr int rows_per_block = 1;

    __device__ auto handle_row(const u16* in, u16* out) -> void
    {
        int lane = threadIdx.x;
        assert_((0 <= lane) && (lane < 32));

        constexpr int S = 256 + 8;

        __shared__ u16 sm[S * 16];

        load_rot_8<dtype, 16, 7, S>(in, sm, lane);

        __syncwarp();

#pragma unroll
        for (int i = 0; i < 16; i++) {
            u16* base = sm + 16 * i;

            constexpr array<int, 8> strides = {1, 2, 4, 8, S, S * 2, S * 4, S * 8};

            // axis 4, 5, 6, 7 needs rotation
            // -> need to be in position 0, 1, 5, 6
            auto in_reg = Frag<
                dtype,
                8,
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

            Frag<dtype, 8, Perm<8>{1, 2, 4, 6, 5, 3, 7, 0}> out_reg =
                apply_on_local_prefix<7>(in_reg, rotate_4<dtype>, lane);

            out_reg.template store<strides, SmBankCount<1>>(base, lane);
        }

        for (int i = 0; i < 16; i++) {
            const u16* ptr_sm = sm + S * i;
            u16* ptr_out = out + 256 * i;

            cp_16(ptr_out + lane * 8, ptr_sm + lane * 8);
        }
    }
};

// had dimension is 2**N
template <DType dtype, int N>
__global__ auto hadamard_transform_ker(const u16* a, u16* out, int num_rows) -> void
{
    RowHandler<dtype, N> handler{};

    constexpr int R = handler.rows_per_block;

#pragma unroll
    for (int j = 0; j < R; j++) {
        int i = blockIdx.x * R + j;

        const u16* a_i = a + i * (1 << N);
        u16* out_i = out + i * (1 << N);

        handler.handle_row(a_i, out_i);
    }

    // for (int i = blockIdx.x; i < num_rows; i += gridDim.x) {
    //     const u16* a_i = a + i * (1 << N);
    //     u16* out_i = out + i * (1 << N);

    //     handler.handle_row(a_i, out_i);
    // }
}

template <torch::ScalarType dtype, int N>
auto run_fht_helper(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t num_rows,
    cudaStream_t stream
) -> void
{
    RowHandler<dtype, N> handler{};

    TORCH_CHECK(num_rows % handler.rows_per_block == 0);

    hadamard_transform_ker<dtype, N>
        <<<num_rows / handler.rows_per_block, 32, 0, stream>>>(
            static_cast<const u16*>(a_mat_ptr),
            static_cast<u16*>(out_ptr),
            num_rows
        );
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
    // TORCH_CHECK(had_size == 256, "This implementation only supports Hadamard size
    // 256");

    TORCH_CHECK(numel % had_size == 0);
    int num_rows = numel / had_size;

    // constexpr int R = 4;
    // TORCH_CHECK(num_rows % R == 0);

    if (had_size == 256) {
        run_fht_helper<dtype, 8>(a_mat_ptr, out_ptr, num_rows, stream);
    } else if (had_size == 4096) {
        run_fht_helper<dtype, 12>(a_mat_ptr, out_ptr, num_rows, stream);
    }

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
// template auto run_fht<torch::ScalarType::BFloat16>(
//     void* a_mat_ptr,
//     void* out_ptr,
//     uint32_t numel,
//     uint32_t had_size,
//     cudaStream_t stream
// ) -> void;
