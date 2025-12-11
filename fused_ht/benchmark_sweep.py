#!/usr/bin/env python3
"""
Benchmark fusion_pipelined.cu across different problem sizes.

This script:
1. Modifies KDIM/NDIM/MDIM constants in fusion_pipelined.cu
2. Compiles with nvcc
3. Runs benchmarks
4. Collects and displays results
"""

import subprocess
import re
import os
import sys
from pathlib import Path
import json

# Test configurations: (KDIM, NDIM, MDIM)
CONFIGS = [
    # Small sizes
    (256, 1024, 1024),
    (256, 2048, 2048),
    (256, 4096, 4096),

    # Medium sizes
    (512, 1024, 1024),
    (512, 2048, 2048),
    (512, 4096, 4096),

    # Large sizes (original)
    (1024, 1024, 1024),
    (1024, 2048, 2048),
    (1024, 4096, 4096),

    # Extra large
    (2048, 1024, 1024),
    (2048, 2048, 2048),
    (2048, 4096, 4096),

    # Huge
    (4096, 1024, 1024),
    (4096, 2048, 2048),
    (4096, 4096, 4096),
]

# Only test fusion_pipelined
KERNEL_FILE = 'fusion_pipelined.cu'

def modify_cuda_file(source_file, output_file, kdim, ndim, mdim):
    """Modify CUDA source to use specified dimensions."""
    with open(source_file, 'r') as f:
        content = f.read()

    # Replace the constexpr dimensions
    content = re.sub(
        r'constexpr int KDIM\s*=\s*\d+;',
        f'constexpr int KDIM   = {kdim};',
        content
    )
    content = re.sub(
        r'constexpr int NDIM\s*=\s*\d+;',
        f'constexpr int NDIM   = {ndim};',
        content
    )
    content = re.sub(
        r'constexpr int MDIM\s*=\s*\d+;',
        f'constexpr int MDIM   = {mdim};',
        content
    )

    with open(output_file, 'w') as f:
        f.write(content)

def compile_cuda(source_file, output_binary):
    """Compile CUDA source file."""
    cmd = [
        'nvcc',
        '-O3',
        '-arch=sm_80',
        '--use_fast_math',
        '-o', output_binary,
        source_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed for {source_file}:")
        print(result.stderr)
        return False
    return True

def run_benchmark(binary):
    """Run benchmark and extract results."""
    result = subprocess.run([binary], capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"Execution failed:")
        print(result.stderr)
        return None

    output = result.stdout

    # Extract timing information
    baseline_match = re.search(r'Baseline.*?min time:\s*([\d.]+)\s*ms', output)
    optimized_match = re.search(r'(?:Optimized|Pipelined).*?min time:\s*([\d.]+)\s*ms', output)
    speedup_match = re.search(r'Speedup.*?:\s*([\d.]+)x', output)
    error_match = re.search(r'Max error\s*=\s*([\d.e+-]+)', output)

    if not (baseline_match and optimized_match):
        print("Could not parse benchmark output:")
        print(output)
        return None

    return {
        'baseline_ms': float(baseline_match.group(1)),
        'optimized_ms': float(optimized_match.group(1)),
        'speedup': float(speedup_match.group(1)) if speedup_match else 0.0,
        'max_error': float(error_match.group(1)) if error_match else 0.0,
        'output': output
    }

def compute_metrics(kdim, ndim, mdim, time_ms):
    """Compute GFLOPS and bandwidth."""
    # FLOPs for Hadamard: KDIM * log2(KDIM) * NDIM additions/subtractions (2 ops per butterfly)
    ht_flops = kdim * (kdim.bit_length() - 1) * ndim * 2

    # FLOPs for GEMM: 2 * MDIM * NDIM * KDIM (multiply-add)
    gemm_flops = 2 * mdim * ndim * kdim

    total_flops = ht_flops + gemm_flops
    gflops = (total_flops / 1e9) / (time_ms / 1000.0)

    # Bandwidth (bytes read/written)
    # Read: X (KDIM*NDIM), W (MDIM*KDIM)
    # Write: C (MDIM*NDIM)
    bytes_rw = (kdim * ndim + mdim * kdim + mdim * ndim) * 4  # float32
    bandwidth_gbps = (bytes_rw / 1e9) / (time_ms / 1000.0)

    return gflops, bandwidth_gbps

def main():
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    if not os.path.exists(KERNEL_FILE):
        print(f"Error: {KERNEL_FILE} not found")
        sys.exit(1)

    results = []

    print("=" * 80)
    print(f"BENCHMARKING: {KERNEL_FILE}")
    print("=" * 80)
    print()

    for kdim, ndim, mdim in CONFIGS:
        config_str = f"KDIM={kdim:4d}, NDIM={ndim:4d}, MDIM={mdim:4d}"
        print(f"{config_str}...", end=' ', flush=True)

        # Modify source
        temp_source = f"temp_{KERNEL_FILE}"
        modify_cuda_file(KERNEL_FILE, temp_source, kdim, ndim, mdim)

        # Compile
        temp_binary = "temp_bench"
        if not compile_cuda(temp_source, temp_binary):
            print("COMPILATION FAILED")
            os.remove(temp_source)
            continue

        # Run benchmark
        try:
            bench_result = run_benchmark(f"./{temp_binary}")
            if bench_result is None:
                print("EXECUTION FAILED")
                continue

            # Compute metrics
            base_gflops, base_bw = compute_metrics(kdim, ndim, mdim, bench_result['baseline_ms'])
            opt_gflops, opt_bw = compute_metrics(kdim, ndim, mdim, bench_result['optimized_ms'])

            result = {
                'kdim': kdim,
                'ndim': ndim,
                'mdim': mdim,
                'baseline_ms': bench_result['baseline_ms'],
                'optimized_ms': bench_result['optimized_ms'],
                'speedup': bench_result['speedup'],
                'baseline_gflops': base_gflops,
                'optimized_gflops': opt_gflops,
                'baseline_bw_gbps': base_bw,
                'optimized_bw_gbps': opt_bw,
                'max_error': bench_result['max_error']
            }
            results.append(result)

            print(f"Base: {bench_result['baseline_ms']:6.2f}ms, "
                  f"Opt: {bench_result['optimized_ms']:6.2f}ms, "
                  f"Speedup: {bench_result['speedup']:.2f}x, "
                  f"GFLOPS: {opt_gflops:.1f}")

        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            # Cleanup
            if os.path.exists(temp_source):
                os.remove(temp_source)
            if os.path.exists(temp_binary):
                os.remove(temp_binary)

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'KDIM':>6} {'NDIM':>6} {'MDIM':>6} {'Base(ms)':>10} {'Opt(ms)':>10} "
          f"{'Speedup':>8} {'GFLOPS':>8} {'BW(GB/s)':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r['kdim']:6d} {r['ndim']:6d} {r['mdim']:6d} "
              f"{r['baseline_ms']:10.2f} {r['optimized_ms']:10.2f} "
              f"{r['speedup']:8.2f} {r['optimized_gflops']:8.1f} "
              f"{r['optimized_bw_gbps']:10.1f}")

    # Save to JSON
    output_file = 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Generate CSV
    csv_file = 'benchmark_results.csv'
    with open(csv_file, 'w') as f:
        f.write("KDIM,NDIM,MDIM,Baseline_ms,Optimized_ms,Speedup,Baseline_GFLOPS,Optimized_GFLOPS,Baseline_BW_GBps,Optimized_BW_GBps,Max_Error\n")
        for r in results:
            f.write(f"{r['kdim']},{r['ndim']},{r['mdim']},"
                   f"{r['baseline_ms']},{r['optimized_ms']},{r['speedup']},"
                   f"{r['baseline_gflops']},{r['optimized_gflops']},"
                   f"{r['baseline_bw_gbps']},{r['optimized_bw_gbps']},{r['max_error']}\n")
    print(f"CSV saved to: {csv_file}")

if __name__ == '__main__':
    main()
