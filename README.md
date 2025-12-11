# Fast Hadamard Transform with Operator Fusion

High-performance CUDA implementation of Hadamard Transform combined with matrix multiplication, demonstrating advanced GPU optimization techniques including kernel fusion and pipelined execution.

## Overview

This project implements and benchmarks several optimizations for the computation:
```
C = W * H(X)
```
where `H(X)` is the Fast Walsh-Hadamard Transform applied to columns of X.

### Key Implementations

1. **Baseline (Unfused)**: Separate Hadamard transform kernel + cuBLAS GEMM
2. **Basic Fusion**: Combined Hadamard + GEMV with warp reductions
3. **Pipelined Fusion**: Advanced implementation with double buffering and cp.async

### Optimization Techniques

- ✅ Operator Fusion (eliminates intermediate matrices)
- ✅ Warp-level parallelism with shuffle reductions
- ✅ Vectorized memory loads (float4)
- ✅ Pipelined execution with double buffering
- ✅ Asynchronous memory transfers (cp.async on Ampere+ GPUs)
- ✅ Parallel Hadamard transform (all threads cooperate)

## Requirements

- CUDA Toolkit (11.0+)
- NVIDIA GPU with compute capability 7.0+ (8.0+ recommended for cp.async)
- Python 3.8+ (for benchmarking scripts)
- cuBLAS library

## Build

### Quick Build

Build the entire project with:

```bash
./rebuild.sh
```

### Manual Build

Build individual CUDA programs:

```bash
# Basic fusion benchmark
cd fused_ht
nvcc -arch=sm_80 -O3 -lcublas fwht_gemv_fusion.cu -o fwht_gemv_fusion

# Pipelined fusion benchmark
nvcc -arch=sm_80 -O3 fusion_pipelined.cu -o fusion_pipelined
```

**Note**: Adjust `-arch=sm_80` to match your GPU architecture:
- sm_75: Turing (RTX 20 series, Tesla T4)
- sm_80: Ampere (A100, RTX 30 series)
- sm_86: Ampere (RTX 30 series mobile)
- sm_89: Ada Lovelace (RTX 40 series)
- sm_90: Hopper (H100)

Check your GPU compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Running the Code

### 1. Basic Fusion Benchmark

Runs correctness tests and performance benchmarks for the basic fused kernel:

```bash
cd fused_ht
./fwht_gemv_fusion
```

**Expected Output:**
```
=== CORRECTNESS TESTS ===
Testing M=32, N=64, B=8...
✓ All tests passed!

=== PERFORMANCE BENCHMARKS ===
Benchmark: M=256, N=256, B=512
Unfused time:        1.234 ms (123.45 GFLOPS)
Fused time:          0.892 ms (170.89 GFLOPS) - Speedup: 1.38x
Fused (vectorized):  0.756 ms (201.32 GFLOPS) - Speedup: 1.63x
Memory saved:        0.512 GB (33.3% reduction)
```

### 2. Pipelined Fusion Benchmark

Runs the advanced pipelined implementation with double buffering:

```bash
cd fused_ht
./fusion_pipelined
```

**Expected Output:**
```
========================================
FUSION + PIPELINING BENCHMARK
========================================
Config: KDIM=1024 NDIM=4096 MDIM=4096

Baseline (Parallel HT + Unfused GEMM) min time: 12.345 ms
Pipelined (Parallel HT + Fused GEMM) min time: 8.123 ms
Speedup from fusion+pipelining: 1.52x

Max error = 1.234e-06 ✓ PASS
```

### 3. Comprehensive Benchmark Sweep

Automatically benchmarks across multiple problem sizes:

```bash
cd fused_ht
python benchmark_sweep.py
```

This script:
- Tests various KDIM/NDIM/MDIM configurations
- Automatically modifies, compiles, and runs benchmarks
- Saves results to `benchmark_results.json` and `benchmark_results.csv`

**Output Files:**
- `benchmark_results.json` - Detailed results with all metrics
- `benchmark_results.csv` - Summary for easy viewing

View results:
```bash
cat fused_ht/benchmark_results.csv
```

### 4. Visualize Results

Generate performance plots:

```bash
cd fused_ht
python plot_results.py
```

**Generated Plots** (saved to `plots/`):
- Speedup vs problem size
- Absolute performance comparison
- Memory bandwidth utilization
- Scaling characteristics

## Project Structure

```
.
├── fused_ht/
│   ├── fusion_pipelined.cu      # Advanced pipelined fusion kernel
│   ├── fwht_gemv_fusion.cu      # Basic fusion kernel with benchmarks
│   ├── benchmark_sweep.py       # Automated benchmarking script
│   ├── plot_results.py          # Visualization script
│   ├── benchmark_results.json   # Detailed benchmark results
│   ├── benchmark_results.csv    # Summary results
│   └── plots/                   # Generated performance plots
├── csrc/
│   ├── main.cu                  # Main Hadamard transform library
│   ├── hada_handler.cu          # Hadamard transform handler
│   └── setup.py                 # Python bindings setup
├── HadaCore/                    # Core Hadamard implementations
├── test/                        # Unit tests
└── README.md                    # This file
```

## Implementation Details

### Basic Fusion (`fwht_gemv_fusion.cu`)

Three kernel implementations:
1. **Unfused**: Separate Hadamard + cuBLAS (baseline)
2. **Fused**: Combined kernel with warp reductions
3. **Fused Vectorized**: Enhanced with float4 loads

Key features:
- Warp-level shuffle reductions (no shared memory for reductions)
- Memory bandwidth reduction (eliminates intermediate RX matrix)
- Suitable for small to medium batch sizes

### Pipelined Fusion (`fusion_pipelined.cu`)

Advanced implementation features:
- **Double Buffering**: Two shared memory buffers (sX0, sX1)
- **Pipelined Schedule**:
  ```
  Load(t+1) || GEMM(t) → HT(t+1)
  ```
- **cp.async**: Asynchronous memory transfers on Ampere+ GPUs
- **Multi-tile Processing**: Each block handles multiple tiles
- **Parallel Hadamard**: All threads cooperate on transform

Shared memory layout:
```
[sX0: KDIM×TILE_N] [sX1: KDIM×TILE_N] [sY: BLOCK_K×TILE_N] [sW: BLOCK_M×BLOCK_K]
```

## Performance Expectations

**Speedup vs Unfused Baseline:**
- Small problems (256×256): ~1.5-2.0x
- Medium problems (1024×1024): ~1.3-1.5x
- Large problems (4096×4096): ~1.2-1.4x

**Why fusion helps:**
1. Eliminates intermediate Yht matrix (~33% memory reduction)
2. Better cache/shared memory utilization
3. Reduced kernel launch overhead
4. Pipelining hides memory latency

**Performance varies based on:**
- GPU memory bandwidth
- Problem size and shape (M, N, K dimensions)
- GPU architecture (Ampere+ benefits from cp.async)

## Troubleshooting

### Compilation Issues

**Error: "No such file or directory: cublas.h"**
```bash
# Ensure CUDA paths are set
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Error: "Unsupported architecture 'sm_80'"**
- Your nvcc version may not support your GPU
- Use an older architecture flag (e.g., `-arch=sm_75`)

### Runtime Issues

**Error: "cp.async.wait_all not supported"**
- Your GPU doesn't support cp.async (requires sm_80+)
- The code automatically falls back to standard loads

**Error: "cudaErrorInvalidDeviceFunction"**
- Kernel was compiled for wrong architecture
- Recompile with correct `-arch` flag

**Poor Performance**
- Check GPU clock speeds: `nvidia-smi`
- Ensure GPU is not throttling due to temperature
- Verify no other processes using GPU

## Customization

### Changing Problem Sizes

Edit the constants in the CUDA files:

```cuda
// In fusion_pipelined.cu or fwht_gemv_fusion.cu
constexpr int KDIM = 1024;  // Hadamard dimension (must be power of 2)
constexpr int NDIM = 4096;  // Number of columns
constexpr int MDIM = 4096;  // Number of rows
```

### Tuning Parameters

Adjust tiling parameters for your GPU:

```cuda
constexpr int TILE_N  = 4;              // Columns per tile
constexpr int BLOCK_M = 128;            // Threads per block
constexpr int BLOCK_K = 32;             // K-dimension tile size
constexpr int TILES_PER_BLOCK = 4;      // Multi-tile processing
```

## References

- Fast Walsh-Hadamard Transform: O(N log N) butterfly algorithm
- Operator Fusion: Combining multiple kernels to reduce memory traffic
- cp.async: NVIDIA Ampere asynchronous memory copy
- Warp-level primitives: `__shfl_down_sync` for efficient reductions


## Authors

Utkarsh (utkarsh5@mit.edu) and Alan Chen (chenxy@mit.edu)
## Acknowledgments

- CUDA programming guides and best practices
- cuBLAS library for high-performance BLAS operations
