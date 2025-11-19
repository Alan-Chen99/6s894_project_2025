#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>

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
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES)
    );
}
