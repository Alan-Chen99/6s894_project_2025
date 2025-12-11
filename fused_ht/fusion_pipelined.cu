/**
 * Pipelined Hadamard Transform + GEMM Fusion
 *
 * This implementation combines operator fusion with pipelined execution to maximize
 * performance. It demonstrates advanced GPU optimization techniques including:
 *
 * Key Optimizations:
 *   1. Parallel Hadamard Transform - All threads cooperate on each column
 *   2. Multi-tile Pipelining - Each block processes multiple tiles with double buffering
 *   3. Asynchronous Memory Transfers - Uses cp.async on Ampere+ GPUs (sm_80)
 *   4. Overlapped Computation - Load(t+1) overlaps with GEMM(t)
 *
 * Pipeline Structure:
 *   Load tile 0 → HT(0) → [Load(1) || GEMM(0)] → HT(1) → [Load(2) || GEMM(1)] → ...
 *
 * Compilation:
 *   nvcc -arch=sm_80 -O3 ./fused_ht/fusion_pipelined.cu -o test_pipe
 *
 * For older GPUs (pre-Ampere):
 *   nvcc -arch=sm_75 -O3 ./fused_ht/fusion_pipelined.cu -o test_pipe
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

/**
 * CUDA error checking macro
 */
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

// ============================================================================
// PROBLEM DIMENSIONS AND TILING PARAMETERS
// ============================================================================

// Matrix dimensions: C[M x N] = W[M x K] * H(X[K x N])
constexpr int KDIM   = 1024;  // Hadamard dimension (must be power of 2)
constexpr int NDIM   = 4096;  // Number of columns in X
constexpr int MDIM   = 4096;  // Number of rows in W

// Tiling configuration
constexpr int TILE_N  = 4;              // Columns per tile
constexpr int BLOCK_M = 128;            // Threads per block (rows processed)
constexpr int BLOCK_K = 32;             // K-dimension tile size for GEMM
constexpr int TILES_PER_BLOCK = 4;      // Number of tiles each block processes

// Compile-time validation
static_assert((KDIM & (KDIM - 1)) == 0, "KDIM must be power of 2");
static_assert(KDIM % BLOCK_K == 0, "KDIM must be divisible by BLOCK_K");

// ============================================================================
// HADAMARD TRANSFORM IMPLEMENTATIONS
// ============================================================================

/**
 * Parallel Hadamard Transform (Thread-Cooperative)
 *
 * All threads in the block cooperate to compute the Hadamard transform.
 * This is more efficient than serial implementations when NN is large.
 *
 * Algorithm: Performs log2(NN) butterfly stages, with all threads working
 * together on each stage. Each butterfly operation computes:
 *   v[i] = v[i] + v[j]
 *   v[j] = v[i] - v[j]
 *
 * @tparam NN      Vector size (must be power of 2)
 * @param v        Shared memory array to transform (size NN)
 * @param tid      Thread ID within block
 * @param threads  Total number of threads in block
 */
template<int NN>
__device__ void hadamard_parallel(float* v, int tid, int threads) {
    // Perform log2(NN) butterfly stages
    for (int h = 1; h < NN; h <<= 1) {
        int total_butterflies = NN / 2;

        // Each thread processes multiple butterflies in a strided pattern
        for (int idx = tid; idx < total_butterflies; idx += threads) {
            int block = idx / h;
            int offset = idx % h;
            int i = block * (2 * h) + offset;
            int j = i + h;

            // Butterfly operation
            float x = v[i];
            float y = v[j];
            v[i] = x + y;
            v[j] = x - y;
        }
        __syncthreads();
    }
}

/**
 * Standard Hadamard Transform (Serial, Single-Thread)
 *
 * Classic sequential implementation for comparison. Only one thread executes
 * this function per column. Less efficient than parallel version for large NN.
 *
 * @tparam NN  Vector size (must be power of 2)
 * @param v    Array to transform (size NN)
 */
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

// ============================================================================
// BASELINE KERNELS (UNFUSED)
// ============================================================================

/**
 * Baseline Hadamard Transform Kernel (Unfused)
 *
 * Applies Hadamard transform to columns of X. Uses parallel Hadamard for
 * fair comparison with the fused kernel. Each block processes TILE_N columns.
 *
 * This is the first stage of the unfused approach:
 *   Stage 1: Yht = H(X)  [this kernel]
 *   Stage 2: C = W * Yht [separate GEMM kernel]
 *
 * @param X    Input matrix (N x K, column-major)
 * @param Yht  Output matrix after Hadamard transform (N x K, column-major)
 */
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

/**
 * Baseline GEMM Kernel (Unfused)
 *
 * Simple matrix multiplication: C = W * Yht
 * Each thread computes one output element using a dot product.
 *
 * This is the second stage of the unfused approach:
 *   Stage 1: Yht = H(X)  [separate Hadamard kernel]
 *   Stage 2: C = W * Yht [this kernel]
 *
 * @param W    Weight matrix (M x K, row-major)
 * @param Yht  Hadamard-transformed input (K x N, column-major)
 * @param C    Output matrix (M x N, row-major)
 */
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

    // Compute dot product
    float acc = 0.f;
    for (int k = 0; k < KDIM; ++k) {
        acc += w_row[k] * y_col[k];
    }

    C[row * NDIM + col] = acc;
}

// ============================================================================
// ASYNCHRONOUS MEMORY COPY (cp.async for Ampere+ GPUs)
// ============================================================================

/**
 * Asynchronous memory copy using cp.async instruction (Ampere+ GPUs, sm_80+)
 *
 * These functions enable overlapping memory transfers with computation.
 * cp.async copies data from global to shared memory asynchronously, allowing
 * the GPU to execute other instructions while the transfer completes.
 *
 * Benefits:
 *   - Overlap memory transfer with computation
 *   - Higher memory bandwidth utilization
 *   - Enables double/triple buffering
 */
#if __CUDA_ARCH__ >= 800

/**
 * Copy 16 bytes from global to shared memory asynchronously
 *
 * @param smem_ptr  Shared memory destination
 * @param gmem_ptr  Global memory source
 */
__device__ __forceinline__
void cp_async_ca_shared_global_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned saddr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 : : "r"(saddr), "l"(gmem_ptr));
}

/**
 * Commit the current group of cp.async operations
 */
__device__ __forceinline__
void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

/**
 * Wait for all pending cp.async operations to complete
 */
__device__ __forceinline__
void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

#endif  // __CUDA_ARCH__ >= 800

// ============================================================================
// FUSED PIPELINED KERNEL
// ============================================================================

/**
 * Fused Hadamard Transform + GEMM with Pipelining
 *
 * This kernel combines Hadamard transform and GEMM in a single fused kernel
 * with advanced pipelining for maximum performance.
 *
 * Key Features:
 *   1. Operator Fusion - Eliminates intermediate Yht matrix from memory
 *   2. Double Buffering - Two shared memory buffers for X tiles (sX0, sX1)
 *   3. Pipelined Execution - Overlaps Load(t+1) with GEMM(t)
 *   4. Parallel Hadamard - All threads cooperate on transform
 *   5. Multi-Tile Processing - Each block processes TILES_PER_BLOCK tiles
 *
 * Pipeline Schedule:
 *   Prologue:  Load(0) → HT(0)
 *   Iteration 0: [Load(1) || GEMM(0)] → HT(1)
 *   Iteration 1: [Load(2) || GEMM(1)] → HT(2)
 *   ...
 *   Iteration n-1: GEMM(n-1)
 *
 * Shared Memory Layout:
 *   sX0 [KDIM * TILE_N]     - First buffer for X tile (double buffering)
 *   sX1 [KDIM * TILE_N]     - Second buffer for X tile (double buffering)
 *   sY  [BLOCK_K * TILE_N]  - Buffer for transformed Y tile
 *   sW  [BLOCK_M * BLOCK_K] - Buffer for W tile
 *
 * @param X  Input matrix (N x K, column-major)
 * @param W  Weight matrix (M x K, row-major)
 * @param C  Output matrix (M x N, row-major)
 */
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

    // ========================================
    // Helper Lambda: Load X Tile
    // ========================================
    /**
     * Loads a tile of X from global to shared memory.
     * Uses cp.async on Ampere+ GPUs for asynchronous transfer.
     *
     * @param tile_n   Tile index (0, 1, 2, ...)
     * @param sX_buf   Destination buffer in shared memory (sX0 or sX1)
     */
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

    // ========================================
    // Helper Lambda: Process Tile (Hadamard + GEMM)
    // ========================================
    /**
     * Processes a tile: applies Hadamard transform and performs GEMM.
     * This function is not used in the optimized pipeline but provided
     * as a reference implementation.
     *
     * @param local_tile  Local tile index within this block (0 to TILES_PER_BLOCK-1)
     * @param tile_n      Global tile index
     * @param sX_buf      Source buffer containing the X tile
     */
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

    // ========================================
    // PIPELINED EXECUTION
    // ========================================
    /**
     * Pipeline Schedule (Fine-Grained):
     *
     * Prologue:
     *   Load(0) → Wait → HT(0)
     *
     * Main Loop (for each tile t):
     *   Start Load(t+1) asynchronously
     *   GEMM(t) using already-transformed data  [overlaps with Load(t+1)!]
     *   Wait for Load(t+1) to complete
     *   HT(t+1)
     *
     * This achieves overlap between memory transfer and computation!
     */
    if (num_tiles == 0) return;

    // ----------------------------------------
    // PROLOGUE: Load and transform tile 0
    // ----------------------------------------
    load_tile(base_tile_n, sX0);
#if __CUDA_ARCH__ >= 800
    cp_async_wait_all();
#endif
    __syncthreads();

    // Apply Hadamard transform to tile 0
    int block_col0 = base_tile_n * TILE_N;
    for (int c = 0; c < TILE_N; ++c) {
        if (block_col0 + c < NDIM) {
            hadamard_parallel<KDIM>(sX0 + c * KDIM, tid, BLOCK_M);
        }
    }

    // ----------------------------------------
    // MAIN PIPELINED LOOP
    // ----------------------------------------
    for (int local_tile = 0; local_tile < num_tiles; ++local_tile) {
        int tile_n = base_tile_n + local_tile;

        // Double buffering: alternate between sX0 and sX1
        float* sX_curr = (local_tile & 1) ? sX1 : sX0;
        float* sX_next = (local_tile & 1) ? sX0 : sX1;

        // ----------------------------------------
        // Step 1: Start loading NEXT tile (asynchronous)
        // ----------------------------------------
        if (local_tile + 1 < num_tiles) {
            load_tile(tile_n + 1, sX_next);
        }

        // ----------------------------------------
        // Step 2: Perform GEMM on CURRENT tile
        // ----------------------------------------
        // This overlaps with the asynchronous load of the next tile!
        if (row_valid) {
            int acc_base = local_tile * TILE_N;
            int curr_col0 = tile_n * TILE_N;

            // Tiled GEMM: Process K dimension in BLOCK_K chunks
            for (int k0 = 0; k0 < KDIM; k0 += BLOCK_K) {
                // Load W tile into shared memory
                for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += BLOCK_M) {
                    int r = idx / BLOCK_K;
                    int kk = idx % BLOCK_K;
                    int row = block_row0 + r;
                    int k = k0 + kk;
                    sW[idx] = (row < MDIM && k < KDIM) ? W[row*KDIM + k] : 0.f;
                }

                // Load Y tile from transformed X data (sX_curr)
                for (int idx = tid; idx < BLOCK_K * TILE_N; idx += BLOCK_M) {
                    int kk = idx / TILE_N;
                    int c = idx % TILE_N;
                    int k = k0 + kk;
                    sY[idx] = (k < KDIM && curr_col0 + c < NDIM)
                              ? sX_curr[c*KDIM + k] : 0.f;
                }
                __syncthreads();

                // Compute partial matrix multiply
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

        // ----------------------------------------
        // Step 3: Wait for next tile load and apply Hadamard transform
        // ----------------------------------------
        if (local_tile + 1 < num_tiles) {
#if __CUDA_ARCH__ >= 800
            cp_async_wait_all();  // Wait for async load to complete
#endif
            __syncthreads();

            // Apply Hadamard transform to the next tile
            // This prepares it for use in the next iteration
            int next_col0 = (tile_n + 1) * TILE_N;
            for (int c = 0; c < TILE_N; ++c) {
                if (next_col0 + c < NDIM) {
                    hadamard_parallel<KDIM>(sX_next + c * KDIM, tid, BLOCK_M);
                }
            }
        }
    }

    // ========================================
    // WRITE RESULTS TO OUTPUT
    // ========================================
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

// ============================================================================
// MAIN BENCHMARK DRIVER
// ============================================================================

/**
 * Main function: benchmarks unfused vs fused+pipelined implementations
 *
 * Benchmarking methodology:
 *   - Both implementations use parallel Hadamard (128 threads) for fair comparison
 *   - Warmup iterations to stabilize GPU state
 *   - Multiple benchmark runs, reporting minimum time
 *   - Correctness verification by comparing outputs
 *
 * Performance expectations:
 *   - Fusion eliminates intermediate Yht matrix (saves memory bandwidth)
 *   - Pipelining overlaps memory transfer with computation
 *   - Combined, these should provide significant speedup over unfused baseline
 */
int main() {
    printf("========================================\n");
    printf("FUSION + PIPELINING BENCHMARK\n");
    printf("========================================\n");
    printf("Config: KDIM=%d NDIM=%d MDIM=%d\n", KDIM, NDIM, MDIM);
    printf("TILE_N=%d, TILES_PER_BLOCK=%d\n", TILE_N, TILES_PER_BLOCK);
    printf("\nBoth use parallel Hadamard (128 threads)\n");
    printf("Comparing: Unfused vs Fused+Pipelined\n");
    printf("========================================\n\n");

    // ========================================
    // Memory Allocation and Initialization
    // ========================================
    size_t bytes_X = KDIM * NDIM * sizeof(float);
    size_t bytes_Yht = KDIM * NDIM * sizeof(float);
    size_t bytes_W = MDIM * KDIM * sizeof(float);
    size_t bytes_C = MDIM * NDIM * sizeof(float);

    // Allocate host memory
    float* h_X   = (float*)malloc(bytes_X);
    float* h_Yht = (float*)malloc(bytes_Yht);
    float* h_W   = (float*)malloc(bytes_W);
    float* h_Cb  = (float*)malloc(bytes_C);  // Baseline output
    float* h_Cf  = (float*)malloc(bytes_C);  // Fused output

    // Initialize with random data
    srand(123);
    for (int j = 0; j < NDIM; ++j) {
        for (int i = 0; i < KDIM; ++i) {
            h_X[j * KDIM + i] = drand48();
        }
    }
    for (int i = 0; i < MDIM * KDIM; ++i) {
        h_W[i] = drand48();
    }

    // Allocate device memory
    float *d_X, *d_Yht, *d_W, *d_Cb, *d_Cf;
    CHECK_CUDA(cudaMalloc(&d_X, bytes_X));
    CHECK_CUDA(cudaMalloc(&d_Yht, bytes_Yht));
    CHECK_CUDA(cudaMalloc(&d_W, bytes_W));
    CHECK_CUDA(cudaMalloc(&d_Cb, bytes_C));
    CHECK_CUDA(cudaMalloc(&d_Cf, bytes_C));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_X, h_X, bytes_X, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W, h_W, bytes_W, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int WARMUP = 2, RUNS = 8;

    // ========================================
    // Benchmark 1: Baseline (Unfused)
    // ========================================
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

    // ========================================
    // Benchmark 2: Fused + Pipelined
    // ========================================
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

    // ========================================
    // Correctness Verification
    // ========================================
    cudaMemcpy(h_Cb, d_Cb, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Cf, d_Cf, bytes_C, cudaMemcpyDeviceToHost);

    // Compare outputs element-wise
    double max_err = 0.0;
    for (int i = 0; i < MDIM * NDIM; ++i) {
        max_err = fmax(max_err, fabs(h_Cb[i] - h_Cf[i]));
    }

    printf("Max error = %.6e ", max_err);
    if (max_err < 1e-3) {
        printf("✓ PASS\n");
    } else {
        printf("✗ FAIL\n");
    }

    // ========================================
    // Cleanup
    // ========================================
    cudaFree(d_X);
    cudaFree(d_Yht);
    cudaFree(d_W);
    cudaFree(d_Cb);
    cudaFree(d_Cf);

    free(h_X);
    free(h_Yht);
    free(h_W);
    free(h_Cb);
    free(h_Cf);

    return 0;
}

/**
 * Expected Results:
 *
 * The fused+pipelined implementation should demonstrate:
 *   1. Reduced memory bandwidth (eliminates intermediate Yht matrix)
 *   2. Better latency hiding through pipelined execution
 *   3. Improved overall throughput compared to baseline
 *
 * Speedup depends on:
 *   - GPU memory bandwidth (fusion helps more on bandwidth-limited cases)
 *   - Problem size (larger matrices benefit more from pipelining)
 *   - GPU architecture (Ampere+ benefits from cp.async)
 */
