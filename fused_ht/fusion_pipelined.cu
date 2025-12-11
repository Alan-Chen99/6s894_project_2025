/*****************************************************
 * fusion_pipelined.cu
 *
 * Combines:
 *  1. Parallel Hadamard (all threads cooperate)
 *  2. Pipelined multi-tile processing
 *
 * Each block processes TILES_PER_BLOCK tiles with pipelining:
 *   Load tile t → Hadamard+GEMM tile t-1 (overlap!)

 nvcc -arch=sm_80 -O3 ./fused_ht/fusion_pipelined.cu -o test_pipe
 *****************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",         \
                    __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                 \
            exit(1);                                          \
        }                                                     \
    } while (0)

constexpr int KDIM   = 1024;
constexpr int NDIM   = 4096;
constexpr int MDIM   = 4096;

constexpr int TILE_N  = 4;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_K = 32;
constexpr int TILES_PER_BLOCK = 4;  // Each block processes 4 tiles

static_assert((KDIM & (KDIM - 1)) == 0, "KDIM must be power of 2");
static_assert(KDIM % BLOCK_K == 0, "KDIM must be divisible by BLOCK_K");

/*****************************************************
 * Parallel Hadamard
 *****************************************************/
template<int NN>
__device__ void hadamard_parallel(float* v, int tid, int threads) {
    for (int h = 1; h < NN; h <<= 1) {
        int total_butterflies = NN / 2;
        for (int idx = tid; idx < total_butterflies; idx += threads) {
            int block = idx / h;
            int offset = idx % h;
            int i = block * (2 * h) + offset;
            int j = i + h;
            float x = v[i];
            float y = v[j];
            v[i] = x + y;
            v[j] = x - y;
        }
        __syncthreads();
    }
}

/*****************************************************
 * Standard Hadamard (baseline)
 *****************************************************/
template<int NN>
__device__ void hadamard_1d(float* v) {
    for (int h = 1; h < NN; h <<= 1) {
        for (int i = 0; i < NN; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                float x = v[j];
                float y = v[j + h];
                v[j]      = x + y;
                v[j + h]  = x - y;
            }
        }
    }
}

/*****************************************************
 * Baseline kernels - Using PARALLEL Hadamard for fair comparison
 *****************************************************/
__global__ void hadamard_columns_baseline(const float* X, float* Yht) {
    extern __shared__ float smem_baseline[];

    int tile_n = blockIdx.x;
    int col_start = tile_n * TILE_N;
    int tid = threadIdx.x;

    if (col_start >= NDIM) return;

    // Load tile into shared memory
    constexpr int TILE_SIZE = KDIM * TILE_N;
    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
        int col_local = i / KDIM;
        int k = i % KDIM;
        int col_global = col_start + col_local;
        if (col_global < NDIM) {
            smem_baseline[i] = X[col_global * KDIM + k];
        }
    }
    __syncthreads();

    // Parallel Hadamard on each column
    for (int c = 0; c < TILE_N; ++c) {
        if (col_start + c < NDIM) {
            hadamard_parallel<KDIM>(smem_baseline + c * KDIM, tid, blockDim.x);
        }
    }

    // Write back
    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
        int col_local = i / KDIM;
        int k = i % KDIM;
        int col_global = col_start + col_local;
        if (col_global < NDIM) {
            Yht[col_global * KDIM + k] = smem_baseline[i];
        }
    }
}

__global__ void gemm_baseline(
    const float* __restrict__ W,
    const float* __restrict__ Yht,
    float* __restrict__ C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= MDIM || col >= NDIM) return;
    const float* w_row = W + row * KDIM;
    const float* y_col = Yht + col * KDIM;
    float acc = 0.f;
    for (int k = 0; k < KDIM; ++k)
        acc += w_row[k] * y_col[k];
    C[row * NDIM + col] = acc;
}

/*****************************************************
 * cp.async
 *****************************************************/
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__
void cp_async_ca_shared_global_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned saddr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 : : "r"(saddr), "l"(gmem_ptr));
}
__device__ __forceinline__
void cp_async_commit_group()  { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__
void cp_async_wait_all()      { asm volatile("cp.async.wait_all;"); }
#endif

/*****************************************************
 * PIPELINED KERNEL
 *
 * Double buffering + parallel Hadamard
 *****************************************************/
__global__ void fused_ht_gemm_pipelined(
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ C
) {
    extern __shared__ float smem[];

    constexpr int TILE_SIZE = KDIM * TILE_N;
    float* sX0 = smem;                       // First buffer
    float* sX1 = sX0 + TILE_SIZE;            // Second buffer (double buffering)
    float* sY  = sX1 + TILE_SIZE;
    float* sW  = sY + BLOCK_K * TILE_N;

    int tid = threadIdx.x;
    int tile_m = blockIdx.y;
    int block_row0 = tile_m * BLOCK_M;
    int row_global = block_row0 + tid;
    bool row_valid = (row_global < MDIM);

    // Base tile for this block
    int base_tile_n = blockIdx.x * TILES_PER_BLOCK;
    int num_tiles = min(TILES_PER_BLOCK, (NDIM + TILE_N - 1) / TILE_N - base_tile_n);

    // Accumulator for all tiles
    float acc[TILES_PER_BLOCK * TILE_N];
    #pragma unroll
    for (int i = 0; i < TILES_PER_BLOCK * TILE_N; ++i) acc[i] = 0.f;

    /*****************************************************
     * Helper: Load X tile
     *****************************************************/
    auto load_tile = [&](int tile_n, float* sX_buf) {
        int block_col0 = tile_n * TILE_N;
        if (block_col0 >= NDIM) return;

        int base_elem = block_col0 * KDIM;

#if __CUDA_ARCH__ >= 800
        constexpr int num_chunks = (TILE_SIZE * 4 + 15) / 16;
        for (int c = tid; c < num_chunks; c += BLOCK_M) {
            int elem = c * 4;
            if (elem < TILE_SIZE && block_col0 + elem/KDIM < NDIM) {
                cp_async_ca_shared_global_16(sX_buf + elem, X + base_elem + elem);
            }
        }
        cp_async_commit_group();
#else
        for (int i = tid; i < TILE_SIZE; i += BLOCK_M) {
            if (block_col0 + i/KDIM < NDIM) {
                sX_buf[i] = X[base_elem + i];
            }
        }
#endif
    };

    /*****************************************************
     * Helper: Hadamard + GEMM on a tile
     *****************************************************/
    auto process_tile = [&](int local_tile, int tile_n, float* sX_buf) {
        int block_col0 = tile_n * TILE_N;
        if (block_col0 >= NDIM) return;

        // Parallel Hadamard
        for (int c = 0; c < TILE_N; ++c) {
            if (block_col0 + c < NDIM) {
                hadamard_parallel<KDIM>(sX_buf + c * KDIM, tid, BLOCK_M);
            }
        }

        // GEMM
        if (!row_valid) return;

        int acc_base = local_tile * TILE_N;
        for (int k0 = 0; k0 < KDIM; k0 += BLOCK_K) {
            // Load W tile
            for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += BLOCK_M) {
                int r = idx / BLOCK_K;
                int kk = idx % BLOCK_K;
                int row = block_row0 + r;
                int k = k0 + kk;
                sW[idx] = (row < MDIM && k < KDIM) ? W[row*KDIM + k] : 0.f;
            }

            // Load Y tile from sX_buf
            for (int idx = tid; idx < BLOCK_K * TILE_N; idx += BLOCK_M) {
                int kk = idx / TILE_N;
                int c = idx % TILE_N;
                int k = k0 + kk;
                sY[idx] = (k < KDIM && block_col0 + c < NDIM)
                          ? sX_buf[c*KDIM + k] : 0.f;
            }
            __syncthreads();

            // Compute
            int r_local = row_global - block_row0;
            #pragma unroll
            for (int kk = 0; kk < BLOCK_K; ++kk) {
                float wv = sW[r_local*BLOCK_K + kk];
                #pragma unroll
                for (int c = 0; c < TILE_N; ++c) {
                    acc[acc_base + c] += wv * sY[kk*TILE_N + c];
                }
            }
            __syncthreads();
        }
    };

    /*****************************************************
     * PIPELINED EXECUTION
     *
     * Fine-grained pipeline:
     *   HT(t) → [Load(t+1) + GEMM(t)] → HT(t+1) → [Load(t+2) + GEMM(t+1)] → ...
     *****************************************************/
    if (num_tiles == 0) return;

    // Prologue: Load and HT tile 0
    load_tile(base_tile_n, sX0);
#if __CUDA_ARCH__ >= 800
    cp_async_wait_all();
#endif
    __syncthreads();

    // HT on tile 0
    int block_col0 = base_tile_n * TILE_N;
    for (int c = 0; c < TILE_N; ++c) {
        if (block_col0 + c < NDIM) {
            hadamard_parallel<KDIM>(sX0 + c * KDIM, tid, BLOCK_M);
        }
    }

    // Main loop
    for (int local_tile = 0; local_tile < num_tiles; ++local_tile) {
        int tile_n = base_tile_n + local_tile;
        float* sX_curr = (local_tile & 1) ? sX1 : sX0;  // Fixed: swapped
        float* sX_next = (local_tile & 1) ? sX0 : sX1;  // Fixed: swapped

        // Start loading NEXT tile (async)
        if (local_tile + 1 < num_tiles) {
            load_tile(tile_n + 1, sX_next);
        }

        // GEMM on current tile (overlaps with load of next tile!)
        if (row_valid) {
            int acc_base = local_tile * TILE_N;
            int curr_col0 = tile_n * TILE_N;

            for (int k0 = 0; k0 < KDIM; k0 += BLOCK_K) {
                // Load W tile
                for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += BLOCK_M) {
                    int r = idx / BLOCK_K;
                    int kk = idx % BLOCK_K;
                    int row = block_row0 + r;
                    int k = k0 + kk;
                    sW[idx] = (row < MDIM && k < KDIM) ? W[row*KDIM + k] : 0.f;
                }

                // Load Y tile
                for (int idx = tid; idx < BLOCK_K * TILE_N; idx += BLOCK_M) {
                    int kk = idx / TILE_N;
                    int c = idx % TILE_N;
                    int k = k0 + kk;
                    sY[idx] = (k < KDIM && curr_col0 + c < NDIM)
                              ? sX_curr[c*KDIM + k] : 0.f;
                }
                __syncthreads();

                // Compute
                int r_local = row_global - block_row0;
                #pragma unroll
                for (int kk = 0; kk < BLOCK_K; ++kk) {
                    float wv = sW[r_local*BLOCK_K + kk];
                    #pragma unroll
                    for (int c = 0; c < TILE_N; ++c) {
                        acc[acc_base + c] += wv * sY[kk*TILE_N + c];
                    }
                }
                __syncthreads();
            }
        }

        // Wait for next tile and do HT on it (overlaps with nothing, but sets up next iteration)
        if (local_tile + 1 < num_tiles) {
#if __CUDA_ARCH__ >= 800
            cp_async_wait_all();
#endif
            __syncthreads();

            // HT on next tile
            int next_col0 = (tile_n + 1) * TILE_N;
            for (int c = 0; c < TILE_N; ++c) {
                if (next_col0 + c < NDIM) {
                    hadamard_parallel<KDIM>(sX_next + c * KDIM, tid, BLOCK_M);
                }
            }
        }
    }

    /*****************************************************
     * Write results
     *****************************************************/
    if (row_valid) {
        for (int local_tile = 0; local_tile < num_tiles; ++local_tile) {
            int tile_n = base_tile_n + local_tile;
            int block_col0 = tile_n * TILE_N;
            int acc_base = local_tile * TILE_N;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                int col = block_col0 + c;
                if (col < NDIM) {
                    C[row_global*NDIM + col] = acc[acc_base + c];
                }
            }
        }
    }
}

/*****************************************************
 * MAIN
 *****************************************************/
int main() {
    printf("========================================\n");
    printf("FUSION + PIPELINING BENCHMARK\n");
    printf("Config: KDIM=%d NDIM=%d MDIM=%d\n", KDIM, NDIM, MDIM);
    printf("TILE_N=%d, TILES_PER_BLOCK=%d\n", TILE_N, TILES_PER_BLOCK);
    printf("\nBoth use parallel Hadamard (128 threads)\n");
    printf("Comparing: Unfused vs Fused+Pipelined\n");
    printf("========================================\n\n");

    size_t bytes_X = KDIM * NDIM * sizeof(float);
    size_t bytes_Yht = KDIM * NDIM * sizeof(float);
    size_t bytes_W = MDIM * KDIM * sizeof(float);
    size_t bytes_C = MDIM * NDIM * sizeof(float);

    float *h_X=(float*)malloc(bytes_X), *h_Yht=(float*)malloc(bytes_Yht);
    float *h_W=(float*)malloc(bytes_W), *h_Cb=(float*)malloc(bytes_C), *h_Cf=(float*)malloc(bytes_C);

    srand(123);
    for (int j=0;j<NDIM;++j)
        for (int i=0;i<KDIM;++i)
            h_X[j*KDIM + i] = drand48();
    for (int i=0;i<MDIM*KDIM;++i)
        h_W[i] = drand48();

    float *d_X, *d_Yht, *d_W, *d_Cb, *d_Cf;
    CHECK_CUDA(cudaMalloc(&d_X, bytes_X));
    CHECK_CUDA(cudaMalloc(&d_Yht, bytes_Yht));
    CHECK_CUDA(cudaMalloc(&d_W, bytes_W));
    CHECK_CUDA(cudaMalloc(&d_Cb, bytes_C));
    CHECK_CUDA(cudaMalloc(&d_Cf, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_X, h_X, bytes_X, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W, h_W, bytes_W, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int WARMUP = 2, RUNS = 8;

    // Baseline with parallel Hadamard
    float min_base = 1e30;
    int grid_baseline = (NDIM + TILE_N - 1) / TILE_N;
    int threads_baseline = BLOCK_M;
    size_t smem_baseline = KDIM * TILE_N * sizeof(float);

    for (int it = 0; it < WARMUP + RUNS; ++it) {
        cudaEventRecord(start);
        hadamard_columns_baseline<<<grid_baseline, threads_baseline, smem_baseline>>>(d_X, d_Yht);
        dim3 b2(16,16);
        dim3 g2((NDIM+15)/16, (MDIM+15)/16);
        gemm_baseline<<<g2,b2>>>(d_W, d_Yht, d_Cb);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms,start,stop);
        if (it>=WARMUP && ms<min_base) min_base=ms;
    }
    printf("Baseline (Parallel HT + Unfused GEMM) min time: %.3f ms\n", min_base);

    // Pipelined
    float min_pipe = 1e30;
    dim3 grid_pipe((NDIM + TILE_N * TILES_PER_BLOCK - 1) / (TILE_N * TILES_PER_BLOCK),
                   (MDIM + BLOCK_M - 1) / BLOCK_M);
    int threads = BLOCK_M;
    size_t smem_bytes = (2*KDIM*TILE_N + BLOCK_K*TILE_N + BLOCK_M*BLOCK_K)*sizeof(float);

    printf("Grid: (%d, %d), Threads: %d\n", grid_pipe.x, grid_pipe.y, threads);
    printf("Shared memory: %.1f KB\n", smem_bytes / 1024.0f);
    printf("Blocks process %d tiles each with pipelining\n\n", TILES_PER_BLOCK);

    CHECK_CUDA(cudaFuncSetAttribute(fused_ht_gemm_pipelined,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    CHECK_CUDA(cudaFuncSetAttribute(fused_ht_gemm_pipelined,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    for (int it = 0; it < WARMUP + RUNS; ++it) {
        cudaEventRecord(start);
        fused_ht_gemm_pipelined<<<grid_pipe, threads, smem_bytes>>>(d_X, d_W, d_Cf);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms,start,stop);
        if (it>=WARMUP && ms<min_pipe) min_pipe=ms;
    }
    printf("Pipelined (Parallel HT + Fused GEMM) min time: %.3f ms\n", min_pipe);
    printf("Speedup from fusion+pipelining: %.2fx\n\n", min_base / min_pipe);

    // Correctness
    cudaMemcpy(h_Cb, d_Cb, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Cf, d_Cf, bytes_C, cudaMemcpyDeviceToHost);
    double max_err = 0.0;
    for (int i=0;i<MDIM*NDIM;++i)
        max_err = fmax(max_err, fabs(h_Cb[i] - h_Cf[i]));
    printf("Max error = %.6e\n", max_err);

    cudaFree(d_X); cudaFree(d_Yht); cudaFree(d_W); cudaFree(d_Cb); cudaFree(d_Cf);
    free(h_X); free(h_Yht); free(h_W); free(h_Cb); free(h_Cf);
    return 0;
}
