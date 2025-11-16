#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <torch/extension.h>

#ifndef __CUDACC__
#define __launch_bounds__(x, y)
#endif

typedef uint32_t b32;
typedef uint16_t b16;

// ----- MMA helpers (kept from before) -----

template <torch::ScalarType dtype>
__device__ __forceinline__ void mma_m16_n8_k16_b16_b16_b16_noacc(
    b32 a0,
    b32 a1,
    b32 a2,
    b32 a3,
    b32 b0,
    b32 b1,
    b32& c0,
    b32& c1
)
{
    static_assert(
        dtype == torch::ScalarType::Half || dtype == torch::ScalarType::BFloat16
    );
    b32 zero = 0;
    if constexpr (dtype == torch::ScalarType::Half) {
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n\t"
            : "=r"(c0), "=r"(c1)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(zero), "r"(zero));
    } else {
        b32 temp0, temp1, temp2, temp3;
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n\t"
            : "=r"(temp0), "=r"(temp1), "=r"(temp2), "=r"(temp3)
            : "r"(a0),
              "r"(a1),
              "r"(a2),
              "r"(a3),
              "r"(b0),
              "r"(b1),
              "r"(zero),
              "r"(zero),
              "r"(zero),
              "r"(zero));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c0) : "r"(temp1), "r"(temp0));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c1) : "r"(temp3), "r"(temp2));
    }
}

template <torch::ScalarType dtype>
__device__ __forceinline__ void mma_m16_n16_k16_b16_b16_b16_noacc(
    b32 a0,
    b32 a1,
    b32 a2,
    b32 a3,
    b32 b0,
    b32 b1,
    b32 b2,
    b32 b3,
    b32& c0,
    b32& c1,
    b32& c2,
    b32& c3
)
{
    mma_m16_n8_k16_b16_b16_b16_noacc<dtype>(a0, a1, a2, a3, b0, b1, c0, c1);
    mma_m16_n8_k16_b16_b16_b16_noacc<dtype>(a0, a1, a2, a3, b2, b3, c2, c3);
}

// Not used for size-256 path but kept so code stays close
__device__ __forceinline__ void matrix_transpose_m8_n8_b16_inplace(b32& a0)
{
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n\t" : "=r"(a0) : "r"(a0));
}

// ----- Hadamard constants (kept from before) -----

#define p_p(i) ((val_1p[i] & 0x0000FFFF) | (b32)val_1p[i] << 16)
#define p_n(i) ((val_1p[i] & 0x0000FFFF) | (b32)val_1n[i] << 16)
#define n_p(i) ((val_1n[i] & 0x0000FFFF) | (b32)val_1p[i] << 16)
#define n_n(i) ((val_1n[i] & 0x0000FFFF) | (b32)val_1n[i] << 16)

template <int log_had_size, torch::ScalarType dtype>
__global__ __launch_bounds__(32, 1) void hadamard_transform_256_kernel(
    const b16* __restrict__ a,
    b16* __restrict__ out,
    int num_rows
)
{
    static_assert(log_had_size == 8, "This demo kernel only handles size 256.");
    static_assert(
        dtype == torch::ScalarType::Half || dtype == torch::ScalarType::BFloat16,
        "Only fp16 and bf16 supported."
    );

    int row = blockIdx.x;
    if (row >= num_rows)
        return;
    int lane = threadIdx.x & 31;

    extern __shared__ b32 smem[]; // 128 x 4B = 512B

    // Base pointers for this row
    const b32* __restrict__ in32 = (const b32*)(a + row * 256);
    b32* __restrict__ out32 = (b32*)(out + row * 256);

// Cooperative load: 256 b16 -> 128 b32
#pragma unroll
    for (int j = 0; j < 4; j++) {
        smem[lane * 4 + j] = in32[lane * 4 + j];
    }
    __syncthreads();

    // Build 16x16 ±1 fragment for 4 stages (we will reuse it twice)
    constexpr b16 fp16_1p[4] =
        {0b0011100110101000, 0b0011100000000000, 0b0011010110101000, 0b0011010000000000};
    constexpr b16 fp16_1n[4] =
        {0b1011100110101000, 0b1011100000000000, 0b1011010110101000, 0b1011010000000000};
    constexpr b16 bf16_1p[4] =
        {0b0011111100110101, 0b0011111100000000, 0b0011111010110101, 0b0011111010000000};
    constexpr b16 bf16_1n[4] =
        {0b1011111100110101, 0b1011111100000000, 0b1011111010110101, 0b1011111010000000};

#define val_type_1p(i) \
    (((dtype) == torch::ScalarType::Half) ? (fp16_1p[i]) : (bf16_1p[i]))
#define val_type_1n(i) \
    (((dtype) == torch::ScalarType::Half) ? (fp16_1n[i]) : (bf16_1n[i]))
    constexpr b16 val_1p[4] =
        {val_type_1p(0), val_type_1p(1), val_type_1p(2), val_type_1p(3)};
    constexpr b16 val_1n[4] =
        {val_type_1n(0), val_type_1n(1), val_type_1n(2), val_type_1n(3)};

    constexpr b32 p_p[4] = {p_p(0), p_p(1), p_p(2), p_p(3)};
    constexpr b32 p_n[4] = {p_n(0), p_n(1), p_n(2), p_n(3)};
    constexpr b32 n_p[4] = {n_p(0), n_p(1), n_p(2), n_p(3)};
    constexpr b32 n_n[4] = {n_n(0), n_n(1), n_n(2), n_n(3)};

    const b32 had_16_p1[4][4] = {
        {0b10001000010001000010001000010001, 0, 0, 0b10001000010001000010001000010001},
        {0b11001100100010000011001100100010, 0, 0, 0b11001100100010000011001100100010},
        {0b11111111101010101100110010011001, 0, 0, 0b11111111101010101100110010011001},
        {0b11111111101010101100110010011001,
         0b11111111101010101100110010011001,
         0b11111111101010101100110010011001,
         0b00000000010101010011001101100110}
    };
    const b32 had_16_p2[4][4] = {
        {0b10000000010000000010000000010000, 0, 0, 0b10000000010000000010000000010000},
        {0b11000000100001000011000000100001, 0, 0, 0b11000000100001000011000000100001},
        {0b11110000101001011100001110010110, 0, 0, 0b11110000101001011100001110010110},
        {0b11110000101001011100001110010110,
         0b11110000101001011100001110010110,
         0b11110000101001011100001110010110,
         0b00001111010110100011110001101001}
    };

    b32 had_frag[8];
// Only i==0 runs because 8 % 4 == 0
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int c_log_h = (i == 0) ? 4 : 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            bool pred1 = (had_16_p1[c_log_h - 1][j] & (1u << (31 - lane))) != 0u;
            bool pred2 = (had_16_p2[c_log_h - 1][j] & (1u << (31 - lane))) != 0u;
            b32 val = pred1 ? (pred2 ? p_p[c_log_h - 1] : p_n[c_log_h - 1])
                            : (pred2 ? n_p[c_log_h - 1] : n_n[c_log_h - 1]);
            had_frag[i * 4 + j] = val;
        }
        if constexpr (log_had_size % 4 == 0)
            break;
    }

    // Load the 16x16 tile from shared into registers (tensor core layout)
    b32 b_frag[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int reg = ((lane & 16) == 0) ? j : (j / 2 * 2 + (1 - j % 2));
        int real_thread_id = (reg == 0 || reg == 2) ? lane : (lane ^ 16);
        int real_row = real_thread_id % 4;
        int real_col = real_thread_id / 4;
        b_frag[j] = smem[(real_row + (reg % 2) * 4) + (real_col + (j / 2) * 8) * 8];
    }
    if ((lane & 16) != 0) {
        b32 t = b_frag[0];
        b_frag[0] = b_frag[1];
        b_frag[1] = t;
        t = b_frag[2];
        b_frag[2] = b_frag[3];
        b_frag[3] = t;
    }
#pragma unroll
    for (int j = 1; j < 4; j += 2) {
        b_frag[j] = __shfl_xor_sync(0xFFFFFFFF, b_frag[j], 16);
    }

    // Do 8 stages in two 4-stage blocks using the same had_frag
    int remaining = 8;
    for (int iter = 0; iter < 2 && remaining > 0; iter++) {
        mma_m16_n16_k16_b16_b16_b16_noacc<dtype>(
            had_frag[0],
            had_frag[1],
            had_frag[2],
            had_frag[3],
            b_frag[0],
            b_frag[1],
            b_frag[2],
            b_frag[3],
            b_frag[0],
            b_frag[1],
            b_frag[2],
            b_frag[3]
        );
        remaining -= 4;
        if (remaining <= 0 && iter == 0) {
            // Unused for size-256 path, left here to mirror original
            matrix_transpose_m8_n8_b16_inplace(b_frag[0]);
            matrix_transpose_m8_n8_b16_inplace(b_frag[1]);
            matrix_transpose_m8_n8_b16_inplace(b_frag[2]);
            matrix_transpose_m8_n8_b16_inplace(b_frag[3]);
        } else {
            b32 t = b_frag[1];
            b_frag[1] = b_frag[2];
            b_frag[2] = t;
        }
    }

// Invert the load swizzle and write back to global
#pragma unroll
    for (int j = 1; j < 4; j += 2) {
        b_frag[j] = __shfl_xor_sync(0xFFFFFFFF, b_frag[j], 16);
    }
    if ((lane & 16) != 0) {
        b32 t = b_frag[0];
        b_frag[0] = b_frag[1];
        b_frag[1] = t;
        t = b_frag[2];
        b_frag[2] = b_frag[3];
        b_frag[3] = t;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int reg = ((lane & 16) == 0) ? j : (j / 2 * 2 + (1 - j % 2));
        int real_thread_id = (reg == 0 || reg == 2) ? lane : (lane ^ 16);
        int real_row = real_thread_id % 4;
        int real_col = real_thread_id / 4;
        out32[(real_row + (reg % 2) * 4) + (real_col + (reg / 2) * 8) * 8] = b_frag[j];
    }
}

// Host wrapper: one block per row, 32 threads per block, 512B smem
template <torch::ScalarType dtype>
void run_fht(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
)
{
    // Only size 256 supported in this demo
    TORCH_CHECK(had_size == (1u << 8), "This demo only supports Hadamard size 256");
    TORCH_CHECK((numel % 256) == 0, "numel must be divisible by 256");

    uint32_t num_rows = numel / 256;
    dim3 grid(num_rows);
    dim3 block(32);
    size_t shmem = 128 * sizeof(b32); // 128 x 4 bytes

    hadamard_transform_256_kernel<8, dtype><<<grid, block, shmem, stream>>>(
        (const b16*)a_mat_ptr,
        (b16*)out_ptr,
        (int)num_rows
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void run_fht<torch::ScalarType::Half>(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
);
template void run_fht<torch::ScalarType::BFloat16>(
    void* a_mat_ptr,
    void* out_ptr,
    uint32_t numel,
    uint32_t had_size,
    cudaStream_t stream
);
