/**
 * Fast 256-point Walsh–Hadamard transform (FHT-256) on Tensor Cores (CUDA).
 *
 * Overview
 * - Dtypes: Half (fp16) and BFloat16 (bf16).
 * - API: PyTorch CUDA extension; call run_fht<dtype>(...) from C++ binding code.
 * - Work partitioning: one warp (32 threads) computes one 256-element row.
 * - Layout: rows of 256 b16 elements (fp16/bf16), contiguous and 4-byte aligned.
 * - Algorithm: two fused 4-stage tensor-core MMA passes with one mid-pass register swap.
 * - Numerics: A-fragment encodes ±0.25 in target dtype to implement Hadamard signs.
 * - Arch:
 *     * fp16 MMA path: SM75+ (Turing and newer).
 *     * bf16 MMA path: SM80+ (Ampere and newer).
 *
 * Boolean magic (Hadamard A-fragment)
 * - Lanes are split into 4 groups of 8 (g = lane >> 3), and sub-lane s in [0..7].
 * - s0, s1, s2 are the bitfields of s. Define F = (g&1 ? s0 : 0) XOR (g&2 ? s1 : 0).
 *   Intuition: F wires which sub-lane bits (s0, s1) feed the sign based on the
 * lane-group.
 * - P1_base = NXOR(F, 0)  -> reproduces 0xFFAACC99 (MSB-first bit order).
 * - P2_base = NXOR(F, s2) -> reproduces 0xF0A5C396 (MSB-first bit order).
 * - Register 3 uses bitwise complement of the base masks (flip all signs).
 *
 * Implementation notes
 * - A-fragment stores ±0.25 in b16x2 registers so one MMA implements four Hadamard stages
 *   at once (like a 4-level butterfly). We run two such passes with a register swap
 * (1<->2) interleaved to complete the full 256-point transform.
 * - B-fragment swizzles ensure the warp loads/stores are coalesced while matching tensor
 *   core operand layouts. Odd registers exchange half-warps via __shfl_xor with mask=16.
 * - Lane masks use "MSB-first indexing": lane 0 reads bit 31, lane 31 reads bit 0.
 *   This matches the original constant encoding and keeps the logic data-independent.
 */

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>

#ifndef __CUDACC__
#define __launch_bounds__(x, y)
#endif

using u32 = uint32_t;
using u16 = uint16_t;

// ----------------------------- Tiny utilities -----------------------------

template <class T> __device__ __forceinline__ auto swap_(T& a, T& b) -> void
{
    T c = a;
    a = b;
    b = c;
}

/**
 * Pack two 16-bit values into one 32-bit word (b16x2).
 * lo -> bits [15:0], hi -> bits [31:16].
 */
__device__ __forceinline__ auto pack_b16x2(u16 lo, u16 hi) -> u32
{
    return static_cast<u32>(lo) | (static_cast<u32>(hi) << 16);
}

/**
 * MSB-first bit test for a warp lane:
 * lane 0 queries bit 31; lane 31 queries bit 0.
 */
__device__ __forceinline__ auto msb_bit_for_lane(u32 mask, int lane) -> bool
{
    return (mask >> (31 - lane)) & 1u;
}

/**
 * Bit encoding of +0.25 and -0.25 in target dtype (fp16/bf16).
 * - For bf16 we encode in the upper 16 bits of a float, following the bf16 layout.
 * - The -0.25 encoding is obtained by flipping the sign bit.
 */
template <torch::ScalarType dtype>
__device__ __forceinline__ auto quarter_pos_bits() -> u16
{
    if constexpr (dtype == torch::ScalarType::Half)
        return static_cast<u16>(0x3400); // +0.25 (fp16)
    else
        return static_cast<u16>(0x3E80); // +0.25 (bf16)
}
template <torch::ScalarType dtype>
__device__ __forceinline__ auto quarter_neg_bits() -> u16
{
    return static_cast<u16>(quarter_pos_bits<dtype>() | 0x8000u);
}

// ---------------------- Hadamard A-fragment mask logic ---------------------

/**
 * Compile-time generator for the 32-bit lane-sign mask used by the A fragment.
 * See the "Boolean magic" overview above for the definition.
 */
constexpr auto nxor_bool(bool a, bool b) -> bool { return a == b; }

constexpr auto had_mask_base(bool include_s2) -> u32
{
    u32 m = 0;
    for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 3; // 4 groups of 8 lanes
        const int s = lane & 7;  // sub-lane in the group
        const bool s0 = static_cast<bool>(s & 1);
        const bool s1 = static_cast<bool>((s >> 1) & 1);
        const bool s2 = static_cast<bool>((s >> 2) & 1);

        // F selects which sub-lane bits contribute based on lane-group
        const bool F = ((g & 1) ? s0 : false) ^ ((g & 2) ? s1 : false);

        // Base bit via NXOR with an optional s2 term
        const bool bit = include_s2 ? nxor_bool(s2, F) : nxor_bool(false, F);

        // MSB-first: lane 0 -> bit 31, lane 31 -> bit 0
        if (bit)
            m |= 1u << (31 - lane);
    }
    return m;
}

// Base (uncomplemented) masks: match original constants but computed constexpr.
static constexpr u32 P1_BASE = had_mask_base(false); // 0xFFAACC99
static constexpr u32 P2_BASE = had_mask_base(true);  // 0xF0A5C396

// ------------------------------ MMA wrappers ------------------------------

/**
 * One Tensor Core MMA for a 16x8x16 multiply with b16 inputs and b16 output regs.
 * D = A x B (+ 0). No accumulation (fresh accumulator = 0).
 *
 * - fp16 path uses f16 accumulators and returns f16 packed as b16x2 regs (u32).
 * - bf16 path accumulates in f32 then packs back to bf16x2 with round-to-nearest.
 *
 * Args
 * - a: 4 regs (b16x2) holding the 16x16 A tile for the fused 4-stage Hadamard signs.
 * - b: 2 regs (b16x2) holding the 16x8 B tile.
 *
 * Returns
 * - 2 regs (b16x2) with the resulting 16x8 D tile.
 */
template <torch::ScalarType dtype>
__device__ __forceinline__ auto mma_m16_n8_k16_b16_b16_b16_noacc(
    const std::array<u32, 4>& a,
    const std::array<u32, 2>& b
) -> std::array<u32, 2>
{
    static_assert(
        dtype == torch::ScalarType::Half || dtype == torch::ScalarType::BFloat16,
        "Only fp16 and bf16 supported."
    );

    constexpr u32 zero = 0;
    u32 c0, c1;

    if constexpr (dtype == torch::ScalarType::Half) {
        // f16 * f16 -> f16 (accum f16), noacc
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                     "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n\t"
                     : "=r"(c0), "=r"(c1)
                     : "r"(a[0]),
                       "r"(a[1]),
                       "r"(a[2]),
                       "r"(a[3]),
                       "r"(b[0]),
                       "r"(b[1]),
                       "r"(zero),
                       "r"(zero));
    } else {
        // bf16 * bf16 -> f32 (accum f32), pack back to bf16x2
        u32 t0, t1, t2, t3;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n\t"
            : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
            : "r"(a[0]),
              "r"(a[1]),
              "r"(a[2]),
              "r"(a[3]),
              "r"(b[0]),
              "r"(b[1]),
              "r"(zero),
              "r"(zero),
              "r"(zero),
              "r"(zero)
        );
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c0) : "r"(t1), "r"(t0));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c1) : "r"(t3), "r"(t2));
    }
    return {c0, c1};
}

/**
 * Two m16n8k16 MMAs to produce a m16n16k16 tile. No accumulation.
 *
 * Args
 * - a: 4 regs (b16x2) for A.
 * - b: 4 regs (b16x2) for B (two 16x8 halves).
 *
 * Returns
 * - 4 regs (b16x2) for D.
 */
template <torch::ScalarType dtype>
__device__ __forceinline__ auto mma_m16_n16_k16_b16_b16_b16_noacc(
    const std::array<u32, 4>& a,
    const std::array<u32, 4>& b
) -> std::array<u32, 4>
{
    const auto c01 = mma_m16_n8_k16_b16_b16_b16_noacc<dtype>(a, {b[0], b[1]});
    const auto c23 = mma_m16_n8_k16_b16_b16_b16_noacc<dtype>(a, {b[2], b[3]});
    return {c01[0], c01[1], c23[0], c23[1]};
}

// ------------------------- B-fragment (data) swizzles ----------------------

/**
 * Warp-coalesced load of a 256-element row (as 128 u32) into shared memory.
 * Each lane copies 4 consecutive u32s (128B per warp; 4B per u32).
 */
__device__ __forceinline__ auto warp_load_256_to_smem(
    const u32* in32,
    u32* smem,
    int lane
) -> void
{
#pragma unroll
    for (int j = 0; j < 4; ++j)
        smem[lane * 4 + j] = in32[lane * 4 + j];
}

/**
 * Build the per-lane B-fragment (four b16x2 regs) from shared memory for m16n16k16 MMA.
 * Includes the XOR-based exchange between half-warps for odd registers.
 *
 * Returns
 * - 4 regs (b16x2) forming the B fragment for this lane.
 *
 * Notes
 * - The mapping computes a "real thread id" that accounts for operand layout
 *   requirements of tensor cores, then pulls the appropriate u32 from smem.
 * - Odd registers exchange between half-warps (lane ^ 16).
 */
__device__ __forceinline__ auto load_B_frag_from_smem(const u32* smem, int lane)
    -> std::array<u32, 4>
{
    std::array<u32, 4> b{};

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // For lanes 16..31, flip (0<->1) and (2<->3) reg order to match MMA operand
        // mapping
        const int reg = (lane & 16) ? (j / 2 * 2 + (1 - j % 2)) : j;

        // For odd regs we pick data as if from the opposite half-warp
        const int real_tid = (reg == 0 || reg == 2) ? lane : (lane ^ 16);

        // The row/col mapping inside 8x8 tiles (see NVIDIA WMMA operand layout)
        const int real_row = real_tid % 4;
        const int real_col = real_tid / 4;

        // Address within smem (8x8 tiles across j/2 and reg%2)
        b[j] = smem[(real_row + (reg % 2) * 4) + (real_col + (j / 2) * 8) * 8];
    }

    // Local shuffle: if in upper half-warp mirrors register pairing
    if (lane & 16) {
        swap_(b[0], b[1]);
        swap_(b[2], b[3]);
    }

    // Cross half-warp exchange for odd regs via XOR lane mask 16
    const unsigned m = __activemask();
#pragma unroll
    for (int j = 1; j < 4; j += 2)
        b[j] = __shfl_xor_sync(m, b[j], 16);

    return b;
}

/**
 * Invert the swizzle performed by load_B_frag_from_smem.
 * Returns a new fragment; the input is not modified.
 */
__device__ __forceinline__ auto invert_B_frag_swizzle(std::array<u32, 4> b, int lane)
    -> std::array<u32, 4>
{
    const unsigned m = __activemask();

    // Undo cross half-warp exchange for odd regs
#pragma unroll
    for (int j = 1; j < 4; j += 2)
        b[j] = __shfl_xor_sync(m, b[j], 16);

    // Undo local register mirroring for upper half-warp
    if (lane & 16) {
        swap_(b[0], b[1]);
        swap_(b[2], b[3]);
    }
    return b;
}

/**
 * Warp-scatter the B-fragment back to global memory matching the original layout.
 * This is the inverse mapping of load_B_frag_from_smem followed by invert_B_frag_swizzle.
 */
__device__ __forceinline__ auto warp_store_B_frag_to_out(
    const std::array<u32, 4>& b,
    u32* out32,
    int lane
) -> void
{
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // Same reg mapping logic as in load, but applied to store address
        const int reg = (lane & 16) ? (j / 2 * 2 + (1 - j % 2)) : j;
        const int real_tid = (reg == 0 || reg == 2) ? lane : (lane ^ 16);
        const int real_row = real_tid % 4;
        const int real_col = real_tid / 4;
        out32[(real_row + (reg % 2) * 4) + (real_col + (reg / 2) * 8) * 8] = b[j];
    }
}

/**
 * Swap registers 1 and 2 of a 4-register fragment in-place.
 * Used between the two fused 4-stage passes, and again to restore order.
 */
template <class Frag> __device__ __forceinline__ auto swap12(Frag& b) -> void
{
    swap_(b[1], b[2]);
}

// ---------------------- A-fragment builder for 4 stages --------------------

/**
 * Build the 4-register A-fragment encoding the Hadamard signs for 4 fused stages.
 * The signs come from P1/P2 masks computed constexpr; register 3 uses complement.
 *
 * Returns
 * - 4 regs (b16x2) with ±0.25 encoded as the target dtype (fp16/bf16).
 */
template <torch::ScalarType dtype>
__device__ __forceinline__ auto build_hadamard_A_frag_4stage(int lane)
    -> std::array<u32, 4>
{
    const u16 qpos = quarter_pos_bits<dtype>();
    const u16 qneg = quarter_neg_bits<dtype>();

    std::array<u32, 4> had{};

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // Reg 3 uses complemented masks (flip all signs)
        const u32 m1 = (j == 3) ? ~P1_BASE : P1_BASE;
        const u32 m2 = (j == 3) ? ~P2_BASE : P2_BASE;

        const u16 lo = msb_bit_for_lane(m1, lane) ? qpos : qneg;
        const u16 hi = msb_bit_for_lane(m2, lane) ? qpos : qneg;

        had[j] = pack_b16x2(lo, hi);
    }
    return had;
}

// --------------------------------- Kernel ---------------------------------

/**
 * Compute one 256-point Hadamard transform per block (one warp), for a batch of rows.
 * grid.x = num_rows, block.x = 32, shared_mem = 128 * sizeof(u32) (512 bytes).
 *
 * Steps per row
 * 1) Coalesced load to shared memory (staging).
 * 2) Assemble B fragment with swizzles and half-warp exchanges.
 * 3) Build A fragment (Hadamard sign pattern, 4 fused stages).
 * 4) Two fused MMA passes with mid-pass swap(1<->2) to complete 8 stages total.
 * 5) Undo swizzles and scatter results to global memory.
 */
template <torch::ScalarType dtype>
__global__ __launch_bounds__(
    32,
    1
) void hadamard_transform_256_kernel(const u16* a, u16* out, int num_rows)
{
    const int row = blockIdx.x;
    if (row >= num_rows)
        return;

    const int lane = threadIdx.x & 31;
    extern __shared__ u32 smem[]; // 128 * 4B = 512B

    // Reinterpret the row as u32 stream for coalesced IO
    const u32* in32 = reinterpret_cast<const u32*>(a + row * 256);
    u32* out32 = reinterpret_cast<u32*>(out + row * 256);

    // 1) Stage to shared memory (warp-coalesced)
    warp_load_256_to_smem(in32, smem, lane);
    __syncthreads();

    // 2) Assemble B fragment (with lane/half-warp swizzles)
    auto b_frag = load_B_frag_from_smem(smem, lane);

    // 3) Build A fragment (Hadamard signs) in registers
    auto had = build_hadamard_A_frag_4stage<dtype>(lane);

    // 4) Two fused 4-stage MMA passes with a mid-pass register swap (1 <-> 2)
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        b_frag = mma_m16_n16_k16_b16_b16_b16_noacc<dtype>(had, b_frag);
        if (pass == 0)
            swap12(b_frag);
    }
    // Restore original register order for store (mirror the single mid-pass swap).
    swap12(b_frag);

    // 5) Undo B-fragment swizzles and scatter results
    b_frag = invert_B_frag_swizzle(b_frag, lane);
    warp_store_B_frag_to_out(b_frag, out32, lane);
}

// --------------------------------- Host -----------------------------------

/**
 * Host entry point: launch the FHT-256 kernel for the given dtype.
 * - a_mat_ptr/out_ptr: device pointers to contiguous b16 buffers.
 * - numel: total number of elements (must be divisible by 256).
 * - had_size: must be 256 for this implementation.
 * - stream: CUDA stream for the launch.
 */
template <torch::ScalarType dtype>
auto run_fht(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
) -> void
{
    TORCH_CHECK(
        had_size == (1u << 8),
        "This implementation only supports Hadamard size 256"
    );
    TORCH_CHECK((numel % 256) == 0, "numel must be divisible by 256");

    const uint32_t num_rows = numel / 256;
    dim3 grid(num_rows), block(32);
    constexpr size_t shmem = 128 * sizeof(u32); // 512B

    hadamard_transform_256_kernel<dtype><<<grid, block, shmem, stream>>>(
        static_cast<const u16*>(a_mat_ptr),
        static_cast<u16*>(out_ptr),
        static_cast<int>(num_rows)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
