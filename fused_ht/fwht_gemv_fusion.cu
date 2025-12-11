/**
 * Fast Walsh-Hadamard Transform + GEMM Fusion
 *
 * This file implements and benchmarks kernel fusion for the Hadamard transform
 * followed by matrix multiplication (GEMM). The fused approach eliminates
 * intermediate memory traffic by computing both operations in a single kernel.
 *
 * Implementations:
 * - Unfused: Separate Hadamard transform kernel + cuBLAS GEMM
 * - Fused: Combined kernel with warp-level reductions
 * - Fused Vectorized: Optimized with float4 vectorized loads

 nvcc -o benchmark ./fused_ht/fwht_gemm_fusion.cu -lcublas -O3
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// ============================================================================
// HADAMARD TRANSFORM KERNELS
// ============================================================================

/**
 * Hadamard Transform Kernel (Unfused Version)
 *
 * Computes the Fast Walsh-Hadamard Transform on each column of the input matrix.
 * Uses shared memory and performs log2(N) butterfly stages.
 *
 * @param X   Input matrix (N x B, column-major)
 * @param RX  Output matrix (N x B, column-major)
 * @param N   Dimension size (must be power of 2)
 * @param B   Batch size (number of columns)
 */
__global__ void hadamard_transform_kernel(
    const float* X,
    float* RX,
    int N,
    int B
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;
    
    extern __shared__ float s_data[];

    // Load input column into shared memory
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        s_data[i] = X[i * B + batch_idx];
    }
    __syncthreads();

    // Fast Walsh-Hadamard Transform: log2(N) butterfly stages
    // Each stage doubles the stride, performing butterfly operations:
    // s[i] = s[i] + s[i + stride]
    // s[i + stride] = s[i] - s[i + stride]
    for (int stride = 1; stride < N; stride *= 2) {
        for (int i = threadIdx.x; i < N / 2; i += blockDim.x) {
            int pair_idx = i / stride;
            int offset = i % stride;
            int idx1 = pair_idx * stride * 2 + offset;
            int idx2 = idx1 + stride;

            if (idx2 < N) {
                float a = s_data[idx1];
                float b = s_data[idx2];
                s_data[idx1] = a + b;
                s_data[idx2] = a - b;
            }
        }
        __syncthreads();
    }

    // Write transformed result back to global memory
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        RX[i * B + batch_idx] = s_data[i];
    }
}

/**
 * Fused Hadamard Transform + GEMV Kernel
 *
 * Combines Hadamard transform and matrix-vector multiplication in a single kernel.
 * The Hadamard transform is computed in shared memory, then results are immediately
 * consumed by GEMV using warp-level shuffle reductions for efficiency.
 *
 * Memory savings: Eliminates the intermediate RX matrix (N x B).
 *
 * @param W  Weight matrix (M x N, row-major)
 * @param X  Input matrix (N x B, column-major)
 * @param Y  Output matrix (M x B, column-major)
 * @param M  Output dimension
 * @param N  Hidden dimension (must be power of 2)
 * @param B  Batch size
 */
__global__ void fused_hadamard_gemv_kernel(
    const float* W,
    const float* X,
    float* Y,
    int M,
    int N,
    int B
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;
    
    extern __shared__ float s_data[];
    
    // Load column into shared memory
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        s_data[i] = X[i * B + batch_idx];
    }
    __syncthreads();
    
    // Hadamard transform: log2(N) stages
    for (int stride = 1; stride < N; stride *= 2) {
        for (int i = threadIdx.x; i < N / 2; i += blockDim.x) {
            int pair_idx = i / stride;
            int offset = i % stride;
            int idx1 = pair_idx * stride * 2 + offset;
            int idx2 = idx1 + stride;
            
            if (idx2 < N) {
                float a = s_data[idx1];
                float b = s_data[idx2];
                s_data[idx1] = a + b;
                s_data[idx2] = a - b;
            }
        }
        __syncthreads();
    }
    
    // GEMV: Y = W * RX (where RX is the transformed data in shared memory)
    // Uses warp-level parallelism and shuffle reductions for efficiency
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;

    for (int row = warp_id; row < M; row += warps_per_block) {
        float sum = 0.0f;

        // Each thread in warp computes partial dot product
        for (int k = lane_id; k < N; k += 32) {
            sum += W[row * N + k] * s_data[k];
        }

        // Warp-level reduction using shuffle operations (no shared memory needed)
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // First thread in warp writes the final result
        if (lane_id == 0) {
            Y[row * B + batch_idx] = sum;
        }
    }
}

/**
 * Fused Hadamard Transform + GEMV Kernel (Vectorized)
 *
 * Enhanced version using float4 vectorized loads for the GEMV portion.
 * Improves memory bandwidth utilization by loading 4 floats per transaction.
 *
 * Requirements: N must be divisible by 4.
 *
 * @param W  Weight matrix (M x N, row-major)
 * @param X  Input matrix (N x B, column-major)
 * @param Y  Output matrix (M x B, column-major)
 * @param M  Output dimension
 * @param N  Hidden dimension (must be power of 2 and divisible by 4)
 * @param B  Batch size
 */
__global__ void fused_hadamard_gemv_kernel_vec(
    const float* W,
    const float* X,
    float* Y,
    int M,
    int N,
    int B
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;
    
    extern __shared__ float s_data[];
    
    // Load column
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        s_data[i] = X[i * B + batch_idx];
    }
    __syncthreads();
    
    // Hadamard transform
    for (int stride = 1; stride < N; stride *= 2) {
        for (int i = threadIdx.x; i < N / 2; i += blockDim.x) {
            int pair_idx = i / stride;
            int offset = i % stride;
            int idx1 = pair_idx * stride * 2 + offset;
            int idx2 = idx1 + stride;
            
            if (idx2 < N) {
                float a = s_data[idx1];
                float b = s_data[idx2];
                s_data[idx1] = a + b;
                s_data[idx2] = a - b;
            }
        }
        __syncthreads();
    }
    
    // GEMV with warp reduction + vectorized loads
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;

    for (int row = warp_id; row < M; row += warps_per_block) {
        float sum = 0.0f;

        // Vectorized loads: each thread loads 4 floats at a time
        // This improves memory bandwidth utilization
        int k = lane_id * 4;
        for (; k + 3 < N; k += 32 * 4) {
            float4 w = *reinterpret_cast<const float4*>(&W[row * N + k]);
            float4 r = *reinterpret_cast<const float4*>(&s_data[k]);
            sum += w.x * r.x + w.y * r.y + w.z * r.z + w.w * r.w;
        }

        // Handle remaining elements (if N is not divisible by 128)
        for (int kk = k; kk < N; kk += 32) {
            sum += W[row * N + kk] * s_data[kk];
        }

        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            Y[row * B + batch_idx] = sum;
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * CUDA error checking macro
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * cuBLAS error checking macro
 */
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Initialize matrix with random values in range [-1, 1]
 */
void init_matrix(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

/**
 * Check if two arrays are element-wise close within tolerance
 *
 * @param a    First array
 * @param b    Second array
 * @param size Number of elements
 * @param tol  Tolerance for difference (default: 1e-3)
 * @return true if all elements match within tolerance
 */
bool check_close(const float* a, const float* b, int size, float tol = 1e-3) {
    float max_diff = 0.0f;
    int mismatch_count = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabs(a[i] - b[i]);
        max_diff = fmax(max_diff, diff);
        if (diff > tol) {
            if (mismatch_count < 5) {  // Only print first 5 mismatches
                printf("Mismatch at index %d: %.6f vs %.6f (diff=%.6f)\n",
                       i, a[i], b[i], diff);
            }
            mismatch_count++;
        }
    }
    if (mismatch_count > 0) {
        printf("Total mismatches: %d/%d\n", mismatch_count, size);
    }
    printf("Max difference: %.6e\n", max_diff);
    return mismatch_count == 0;
}

// ============================================================================
// BENCHMARKING
// ============================================================================

/**
 * Benchmark the unfused approach: separate Hadamard transform + cuBLAS GEMM
 *
 * @return Average execution time in milliseconds
 */
float benchmark_unfused(
    cublasHandle_t handle,
    const float* d_W, const float* d_X, float* d_RX, float* d_Y,
    int M, int N, int B,
    int num_warmup, int num_iters
) {
    const int block_size = 256;
    const int smem_size = N * sizeof(float);
    
    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        hadamard_transform_kernel<<<B, block_size, smem_size>>>(d_X, d_RX, N, B);
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            B, M, N, &alpha, d_RX, B, d_W, N, &beta, d_Y, B));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        hadamard_transform_kernel<<<B, block_size, smem_size>>>(d_X, d_RX, N, B);
        CUDA_CHECK(cudaGetLastError());
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            B, M, N, &alpha, d_RX, B, d_W, N, &beta, d_Y, B));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return ms / num_iters;
}

/**
 * Benchmark the fused approach: combined Hadamard transform + GEMV kernel
 *
 * @param use_vectorized  If true, uses the vectorized kernel (requires N % 4 == 0)
 * @return Average execution time in milliseconds
 */
float benchmark_fused(
    const float* d_W, const float* d_X, float* d_Y,
    int M, int N, int B,
    int num_warmup, int num_iters,
    bool use_vectorized = false
) {
    const int block_size = 256;
    const int smem_size = N * sizeof(float);
    
    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        if (use_vectorized && N % 4 == 0) {
            fused_hadamard_gemv_kernel_vec<<<B, block_size, smem_size>>>(d_W, d_X, d_Y, M, N, B);
        } else {
            fused_hadamard_gemv_kernel<<<B, block_size, smem_size>>>(d_W, d_X, d_Y, M, N, B);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        if (use_vectorized && N % 4 == 0) {
            fused_hadamard_gemv_kernel_vec<<<B, block_size, smem_size>>>(d_W, d_X, d_Y, M, N, B);
        } else {
            fused_hadamard_gemv_kernel<<<B, block_size, smem_size>>>(d_W, d_X, d_Y, M, N, B);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return ms / num_iters;
}

// ============================================================================
// CORRECTNESS TEST
// ============================================================================

/**
 * Test correctness by comparing fused kernel output against unfused reference
 *
 * @param M               Output dimension
 * @param N               Hidden dimension (must be power of 2)
 * @param B               Batch size
 * @param test_vectorized If true, tests the vectorized kernel
 * @return true if outputs match within tolerance
 */
bool test_correctness(int M, int N, int B, bool test_vectorized = false) {
    printf("\n=== Testing Correctness%s: M=%d, N=%d, B=%d ===\n", 
           test_vectorized ? " (Vectorized)" : "", M, N, B);
    
    float* h_W = (float*)malloc(M * N * sizeof(float));
    float* h_X = (float*)malloc(N * B * sizeof(float));
    float* h_Y_unfused = (float*)malloc(M * B * sizeof(float));
    float* h_Y_fused = (float*)malloc(M * B * sizeof(float));
    
    init_matrix(h_W, M * N);
    init_matrix(h_X, N * B);
    
    float *d_W, *d_X, *d_RX, *d_Y_unfused, *d_Y_fused;
    CUDA_CHECK(cudaMalloc(&d_W, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, N * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_RX, N * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y_unfused, M * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y_fused, M * B * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_W, h_W, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * B * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Disable TF32 to force true FP32 computation for fair comparison
    // (TF32 uses reduced precision internally on Ampere+ GPUs)
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    cublasMath_t math_mode;
    CUBLAS_CHECK(cublasGetMathMode(handle, &math_mode));
    printf("cuBLAS Math Mode: ");
    if (math_mode == CUBLAS_TF32_TENSOR_OP_MATH) {
        printf("TF32_TENSOR_OP (Tensor Cores)\n");
    } else if (math_mode == CUBLAS_DEFAULT_MATH) {
        printf("DEFAULT (CUDA Cores)\n");
    } else if (math_mode == CUBLAS_PEDANTIC_MATH) {
        printf("PEDANTIC (True FP32, no TF32)\n");
    } else {
        printf("Other (%d)\n", math_mode);
    }
    
    const int block_size = 256;
    const int smem_size = N * sizeof(float);
    
    // Unfused reference
    hadamard_transform_kernel<<<B, block_size, smem_size>>>(d_X, d_RX, N, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        B, M, N, &alpha, d_RX, B, d_W, N, &beta, d_Y_unfused, B));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Fused version
    if (test_vectorized && N % 4 == 0) {
        fused_hadamard_gemv_kernel_vec<<<B, block_size, smem_size>>>(d_W, d_X, d_Y_fused, M, N, B);
    } else {
        fused_hadamard_gemv_kernel<<<B, block_size, smem_size>>>(d_W, d_X, d_Y_fused, M, N, B);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_Y_unfused, d_Y_unfused, M * B * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y_fused, d_Y_fused, M * B * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = check_close(h_Y_unfused, h_Y_fused, M * B);
    
    free(h_W); free(h_X); free(h_Y_unfused); free(h_Y_fused);
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_RX));
    CUDA_CHECK(cudaFree(d_Y_unfused));
    CUDA_CHECK(cudaFree(d_Y_fused));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return correct;
}

// ============================================================================
// PERFORMANCE BENCHMARK
// ============================================================================

/**
 * Run comprehensive performance comparison for given matrix dimensions
 *
 * Compares unfused, fused, and fused-vectorized implementations.
 * Prints execution time, GFLOPS, speedup, and memory savings.
 *
 * @param M  Output dimension
 * @param N  Hidden dimension (must be power of 2)
 * @param B  Batch size
 */
void run_benchmark(int M, int N, int B) {
    printf("\n=== Benchmark: M=%d, N=%d, B=%d ===\n", M, N, B);
    
    const int num_warmup = 10;
    const int num_iters = 100;
    
    float* h_W = (float*)malloc(M * N * sizeof(float));
    float* h_X = (float*)malloc(N * B * sizeof(float));
    init_matrix(h_W, M * N);
    init_matrix(h_X, N * B);
    
    float *d_W, *d_X, *d_RX, *d_Y_unfused, *d_Y_fused;
    CUDA_CHECK(cudaMalloc(&d_W, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, N * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_RX, N * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y_unfused, M * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y_fused, M * B * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_W, h_W, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * B * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Disable TF32 to force true FP32 computation for fair comparison
    // (TF32 uses reduced precision internally on Ampere+ GPUs)
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    cublasMath_t math_mode;
    CUBLAS_CHECK(cublasGetMathMode(handle, &math_mode));
    printf("cuBLAS Math Mode: ");
    if (math_mode == CUBLAS_TF32_TENSOR_OP_MATH) {
        printf("TF32_TENSOR_OP (Tensor Cores)\n");
    } else if (math_mode == CUBLAS_DEFAULT_MATH) {
        printf("DEFAULT (CUDA Cores)\n");
    } else if (math_mode == CUBLAS_PEDANTIC_MATH) {
        printf("PEDANTIC (True FP32, no TF32)\n");
    } else {
        printf("Other (%d)\n", math_mode);
    }
    
    // ========================================
    // Benchmark all versions
    // ========================================
    float time_unfused = benchmark_unfused(handle, d_W, d_X, d_RX, d_Y_unfused,
                                            M, N, B, num_warmup, num_iters);
    float time_fused = benchmark_fused(d_W, d_X, d_Y_fused,
                                        M, N, B, num_warmup, num_iters, false);
    float time_fused_vec = benchmark_fused(d_W, d_X, d_Y_fused,
                                            M, N, B, num_warmup, num_iters, true);

    // ========================================
    // Compute performance metrics
    // ========================================
    float speedup = time_unfused / time_fused;
    float speedup_vec = time_unfused / time_fused_vec;

    // Memory traffic analysis
    // Unfused: Read W, Read X, Write RX, Read RX, Write Y
    // Fused:   Read W, Read X, Write Y (RX stays in shared memory)
    float mem_traffic_unfused = (M*N + N*B + N*B + N*B + M*B) * sizeof(float) / 1e9;
    float mem_traffic_fused = (M*N + N*B + M*B) * sizeof(float) / 1e9;
    float mem_saved = mem_traffic_unfused - mem_traffic_fused;

    // FLOP counting
    float compute_hadamard = (N * log2f(N) * B);  // log2(N) butterfly stages
    float compute_gemm = (2.0f * M * N * B);      // 2*M*N*B for matrix multiply
    float total_flops = compute_hadamard + compute_gemm;

    // ========================================
    // Print results
    // ========================================
    printf("FLOPs: Hadamard=%.2e, GEMM=%.2e, Total=%.2e\n",
           compute_hadamard, compute_gemm, total_flops);
    printf("Unfused time:        %.3f ms (%.2f GFLOPS)\n",
           time_unfused, (total_flops / 1e9) / (time_unfused / 1000.0f));
    printf("Fused time:          %.3f ms (%.2f GFLOPS) - Speedup: %.2fx\n",
           time_fused, (total_flops / 1e9) / (time_fused / 1000.0f), speedup);
    printf("Fused (vectorized):  %.3f ms (%.2f GFLOPS) - Speedup: %.2fx\n",
           time_fused_vec, (total_flops / 1e9) / (time_fused_vec / 1000.0f), speedup_vec);
    printf("Memory saved:        %.3f GB (%.1f%% reduction)\n",
           mem_saved, 100.0f * mem_saved / mem_traffic_unfused);
    
    free(h_W); free(h_X);
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_RX));
    CUDA_CHECK(cudaFree(d_Y_unfused));
    CUDA_CHECK(cudaFree(d_Y_fused));
    CUBLAS_CHECK(cublasDestroy(handle));
}

// ============================================================================
// MAIN
// ============================================================================

/**
 * Main driver: runs correctness tests and performance benchmarks
 *
 * Tests multiple problem sizes to demonstrate fusion benefits:
 * - Small problems where fusion overhead savings dominate
 * - Medium problems showing balanced benefits
 * - Large problems where cuBLAS optimizations are strong
 */
int main(int argc, char** argv) {
    srand(42);

    // ========================================
    // Correctness Tests
    // ========================================
    printf("=== CORRECTNESS TESTS ===\n");
    assert(test_correctness(32, 64, 8, false));
    assert(test_correctness(64, 128, 16, false));
    assert(test_correctness(128, 256, 32, true));
    printf("\n✓ All correctness tests passed!\n");

    // ========================================
    // Performance Benchmarks
    // ========================================
    printf("\n=== PERFORMANCE BENCHMARKS ===\n");

    printf("\n--- Sweet Spot: Small M, Medium N, Large B ---\n");
    run_benchmark(256, 256, 512);
    run_benchmark(512, 512, 1024);
    run_benchmark(1024, 1024, 2048);

    printf("\n--- Larger M (cuBLAS advantage) ---\n");
    run_benchmark(1024, 1024, 256);
    run_benchmark(2048, 2048, 512);

    printf("\n--- Very Small (Fusion advantage) ---\n");
    run_benchmark(64, 128, 256);
    run_benchmark(128, 256, 512);

    return 0;
}