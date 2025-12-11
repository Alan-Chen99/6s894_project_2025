#!/usr/bin/env python3
"""
Plotting script for benchmark results.

Reads benchmark_results.csv and generates visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_results(csv_file='benchmark_results.csv'):
    """Load benchmark results from CSV."""
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found. Run benchmark_sweep.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_file)
    return df

def plot_speedup_by_size(df, output_dir='plots'):
    """Plot speedup vs problem size."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Group by each dimension
    kdims = sorted(df['KDIM'].unique())
    ndims = sorted(df['NDIM'].unique())
    mdims = sorted(df['MDIM'].unique())

    # Plot 1: Speedup vs KDIM (fix NDIM=MDIM=4096)
    ax = axes[0]
    subset = df[(df['NDIM'] == 4096) & (df['MDIM'] == 4096)]
    if not subset.empty:
        ax.plot(subset['KDIM'], subset['Speedup'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('KDIM', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Speedup vs KDIM (NDIM=MDIM=4096)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

    # Plot 2: Speedup vs NDIM (fix KDIM=1024, MDIM=4096)
    ax = axes[1]
    subset = df[(df['KDIM'] == 1024) & (df['MDIM'] == 4096)]
    if not subset.empty:
        ax.plot(subset['NDIM'], subset['Speedup'], 's-', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('NDIM', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Speedup vs NDIM (KDIM=1024, MDIM=4096)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

    # Plot 3: Speedup vs MDIM (fix KDIM=1024, NDIM=4096)
    ax = axes[2]
    subset = df[(df['KDIM'] == 1024) & (df['NDIM'] == 4096)]
    if not subset.empty:
        ax.plot(subset['MDIM'], subset['Speedup'], '^-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('MDIM', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Speedup vs MDIM (KDIM=1024, NDIM=4096)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_by_dimension.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/speedup_by_dimension.png")
    plt.close()

def plot_gflops_comparison(df, output_dir='plots'):
    """Plot GFLOPS comparison."""
    Path(output_dir).mkdir(exist_ok=True)

    # Select interesting configurations
    configs = [
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (4096, 4096, 4096),
    ]

    labels = []
    baseline_gflops = []
    optimized_gflops = []

    for kdim, ndim, mdim in configs:
        subset = df[(df['KDIM'] == kdim) & (df['NDIM'] == ndim) & (df['MDIM'] == mdim)]
        if not subset.empty:
            row = subset.iloc[0]
            labels.append(f'K={kdim}')
            baseline_gflops.append(row['Baseline_ms'])
            optimized_gflops.append(row['Optimized_ms'])

    if not labels:
        print("No data for GFLOPS comparison")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_gflops, width, label='Baseline (Unfused)', color='#E74C3C')
    bars2 = ax.bar(x + width/2, optimized_gflops, width, label='Fused+Pipelined', color='#2ECC71')

    ax.set_xlabel('Configuration (NDIM=MDIM=4096)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Execution Time Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/time_comparison.png")
    plt.close()

def plot_performance_heatmap(df, output_dir='plots'):
    """Plot performance heatmap for different KDIM configurations."""
    Path(output_dir).mkdir(exist_ok=True)

    # Get unique KDIM values that have data
    kdims_available = sorted(df['KDIM'].unique())

    # Create one heatmap per KDIM value
    for kdim in kdims_available:
        subset = df[df['KDIM'] == kdim]

        if subset.empty:
            continue

        # Get unique NDIM and MDIM values for this KDIM
        ndims = sorted(subset['NDIM'].unique())
        mdims = sorted(subset['MDIM'].unique())

        # Create matrix with NaN for missing values
        matrix = np.full((len(mdims), len(ndims)), np.nan)
        for i, mdim in enumerate(mdims):
            for j, ndim in enumerate(ndims):
                row = subset[(subset['NDIM'] == ndim) & (subset['MDIM'] == mdim)]
                if not row.empty:
                    matrix[i, j] = row['Speedup'].iloc[0]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create custom colormap that shows NaN as gray
        cmap = plt.cm.RdYlGn
        cmap.set_bad(color='lightgray')

        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=1, vmax=5)

        # Set ticks and labels
        ax.set_xticks(range(len(ndims)))
        ax.set_yticks(range(len(mdims)))
        ax.set_xticklabels(ndims, fontsize=11)
        ax.set_yticklabels(mdims, fontsize=11)
        ax.set_xlabel('NDIM', fontsize=13, weight='bold')
        ax.set_ylabel('MDIM', fontsize=13, weight='bold')
        ax.set_title(f'Speedup Heatmap (KDIM={kdim})', fontsize=15, weight='bold', pad=15)

        # Add text annotations
        for i in range(len(mdims)):
            for j in range(len(ndims)):
                if not np.isnan(matrix[i, j]):
                    value = matrix[i, j]
                    # Choose text color based on background
                    text_color = 'white' if value < 2.5 else 'black'
                    ax.text(j, i, f'{value:.2f}×',
                           ha="center", va="center",
                           color=text_color, fontsize=12, weight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Speedup', fraction=0.046, pad=0.04)
        cbar.set_label('Speedup', fontsize=12, weight='bold')

        # Add grid
        ax.set_xticks([x - 0.5 for x in range(1, len(ndims))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(mdims))], minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

        plt.tight_layout()
        filename = f'{output_dir}/speedup_heatmap_kdim{kdim}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    # Also create a combined overview plot
    kdims_to_plot = [k for k in [256, 512, 1024, 2048, 4096] if k in kdims_available]
    if len(kdims_to_plot) > 0:
        nrows = (len(kdims_to_plot) + 2) // 3
        ncols = min(3, len(kdims_to_plot))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))

        if len(kdims_to_plot) == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows * ncols > 1 else [axes]

        for idx, kdim in enumerate(kdims_to_plot):
            ax = axes[idx]
            subset = df[df['KDIM'] == kdim]

            ndims = sorted(subset['NDIM'].unique())
            mdims = sorted(subset['MDIM'].unique())

            matrix = np.full((len(mdims), len(ndims)), np.nan)
            for i, mdim in enumerate(mdims):
                for j, ndim in enumerate(ndims):
                    row = subset[(subset['NDIM'] == ndim) & (subset['MDIM'] == mdim)]
                    if not row.empty:
                        matrix[i, j] = row['Speedup'].iloc[0]

            cmap = plt.cm.RdYlGn
            cmap.set_bad(color='lightgray')
            im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=1, vmax=5)

            ax.set_xticks(range(len(ndims)))
            ax.set_yticks(range(len(mdims)))
            ax.set_xticklabels(ndims, fontsize=9)
            ax.set_yticklabels(mdims, fontsize=9)
            ax.set_xlabel('NDIM', fontsize=10)
            ax.set_ylabel('MDIM', fontsize=10)
            ax.set_title(f'KDIM={kdim}', fontsize=11, weight='bold')

            # Add annotations
            for i in range(len(mdims)):
                for j in range(len(ndims)):
                    if not np.isnan(matrix[i, j]):
                        value = matrix[i, j]
                        text_color = 'white' if value < 2.5 else 'black'
                        ax.text(j, i, f'{value:.2f}',
                               ha="center", va="center",
                               color=text_color, fontsize=8)

        # Remove extra subplots
        for idx in range(len(kdims_to_plot), len(axes)):
            fig.delaxes(axes[idx])

        # Add single colorbar
        fig.colorbar(im, ax=axes, label='Speedup', fraction=0.046, pad=0.04)

        plt.suptitle('Speedup Overview Across All Configurations',
                    fontsize=16, weight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/speedup_heatmap_overview.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/speedup_heatmap_overview.png")
        plt.close()

def plot_bandwidth_utilization(df, output_dir='plots'):
    """Plot bandwidth utilization."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get configurations with NDIM=MDIM=4096
    subset = df[(df['NDIM'] == 4096) & (df['MDIM'] == 4096)].sort_values('KDIM')

    if subset.empty:
        print("No data for bandwidth plot")
        return

    x = range(len(subset))
    labels = [f'K={k}' for k in subset['KDIM']]

    baseline_bw = subset['Baseline_BW_GBps'].values
    optimized_bw = subset['Optimized_BW_GBps'].values

    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], baseline_bw, width,
                   label='Baseline', color='#3498DB')
    bars2 = ax.bar([i + width/2 for i in x], optimized_bw, width,
                   label='Optimized', color='#E67E22')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Configuration (NDIM=MDIM=4096)', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Memory Bandwidth by KDIM', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/bandwidth_utilization.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/bandwidth_utilization.png")
    plt.close()

def plot_summary_table(df, output_dir='plots'):
    """Create a summary table plot."""
    Path(output_dir).mkdir(exist_ok=True)

    # Select key configurations
    configs = [
        (1024, 1024, 1024),
        (1024, 2048, 2048),
        (1024, 4096, 4096),
        (2048, 2048, 2048),
        (2048, 4096, 4096),
        (4096, 4096, 4096),
    ]

    table_data = []
    for kdim, ndim, mdim in configs:
        subset = df[(df['KDIM'] == kdim) & (df['NDIM'] == ndim) & (df['MDIM'] == mdim)]
        if not subset.empty:
            row = subset.iloc[0]
            table_data.append([
                f"{kdim}×{ndim}×{mdim}",
                f"{row['Baseline_ms']:.2f}",
                f"{row['Optimized_ms']:.2f}",
                f"{row['Speedup']:.2f}×",
                f"{row['Optimized_GFLOPS']:.1f}",
            ])

    if not table_data:
        print("No data for summary table")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    headers = ['Config\n(K×N×M)', 'Baseline\n(ms)', 'Optimized\n(ms)', 'Speedup', 'GFLOPS']

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')

    plt.title('Performance Summary', fontsize=16, weight='bold', pad=20)
    plt.savefig(f'{output_dir}/summary_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/summary_table.png")
    plt.close()

def main():
    print("=" * 60)
    print("BENCHMARK RESULTS VISUALIZATION")
    print("=" * 60)
    print()

    # Load data
    df = load_results()
    print(f"Loaded {len(df)} benchmark results\n")

    # Generate plots
    output_dir = 'plots'
    Path(output_dir).mkdir(exist_ok=True)

    print("Generating plots...")
    plot_speedup_by_size(df, output_dir)
    plot_gflops_comparison(df, output_dir)
    plot_performance_heatmap(df, output_dir)
    plot_bandwidth_utilization(df, output_dir)
    plot_summary_table(df, output_dir)

    print()
    print("=" * 60)
    print(f"All plots saved to {output_dir}/")
    print("=" * 60)

if __name__ == '__main__':
    main()
