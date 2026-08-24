#!/usr/bin/env python3
"""Generate dependency-free SVG/CSV artifacts for dynamic RHT training."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


COLORS = {
    "bf16": "#70a5ff",
    "block_fp8": "#ffc46b",
    "dynamic_rht_block_fp8": "#51d6ca",
}
LABELS = {
    "bf16": "BF16",
    "block_fp8": "TE block-FP8",
    "dynamic_rht_block_fp8": "Dynamic RHT + block-FP8",
}


def svg_frame(title: str, subtitle: str, body: str, width=900, height=520) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>
<rect width="100%" height="100%" rx="16" fill="#0f151f"/>
<text x="54" y="42" fill="#edf2fa" font-family="Inter,system-ui,sans-serif" font-size="20" font-weight="700">{html.escape(title)}</text>
<text x="54" y="65" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">{html.escape(subtitle)}</text>
{body}</svg>\n'''


def convergence_svg(payload: dict) -> str:
    curves = payload["curves"]
    width, height = 900, 520
    left, right, top, bottom = 72, 28, 94, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    steps = payload["steps"]
    values = [v for method in curves.values() for v in method["loss"]]
    low = 10 ** math.floor(math.log10(min(values)))
    high = 10 ** math.ceil(math.log10(max(values)))

    def x(index: int) -> float:
        return left + index * plot_w / (steps - 1)

    def y(value: float) -> float:
        return top + (math.log10(high) - math.log10(value)) * plot_h / (
            math.log10(high) - math.log10(low)
        )

    parts = []
    power = math.floor(math.log10(low))
    while 10**power <= high:
        value = 10**power
        py = y(value)
        parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#273246" stroke-dasharray="3 5"/>')
        parts.append(f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" fill="#8e9bb0" font-family="ui-monospace,monospace" font-size="11">10^{power}</text>')
        power += 1
    for step in (1, 20, 40, 60, 80, 100, steps):
        px = x(step - 1)
        parts.append(f'<text x="{px:.2f}" y="{height-25}" text-anchor="middle" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="11">{step}</text>')
    parts.append(f'<text x="{width/2}" y="{height-7}" text-anchor="middle" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">Optimizer step</text>')
    parts.append(f'<text transform="translate(17 {top+plot_h/2}) rotate(-90)" text-anchor="middle" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">MSE loss (log scale)</text>')
    for index, (name, record) in enumerate(curves.items()):
        points = " ".join(
            f"{x(i):.2f},{y(value):.2f}" for i, value in enumerate(record["loss"])
        )
        color = COLORS[name]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"/>')
        lx = 520 + index * 120
        parts.append(f'<line x1="{lx}" y1="43" x2="{lx+22}" y2="43" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+28}" y="47" fill="#cbd5e5" font-family="Inter,system-ui,sans-serif" font-size="11">{html.escape(LABELS[name])}</text>')
    return svg_frame(
        "Dynamic RHT training convergence",
        "Matched teacher–student SwiGLU MLP · H100 NVL · FP32 original-basis AdamW masters",
        "".join(parts),
    )


def bar_svg(title: str, subtitle: str, labels: list[str], values: list[float], unit: str) -> str:
    width, height = 900, 500
    left, right, top, bottom = 90, 35, 95, 92
    plot_h = height - top - bottom
    max_value = max(values) * 1.18
    slot = (width - left - right) / len(values)
    bar_w = min(125, slot * 0.58)
    parts = []
    for tick in range(5):
        value = max_value * tick / 4
        py = top + plot_h * (1 - tick / 4)
        parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#273246" stroke-dasharray="3 5"/>')
        parts.append(f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" fill="#8e9bb0" font-family="ui-monospace,monospace" font-size="11">{value:.2f}</text>')
    palette = ["#70a5ff", "#ffc46b", "#51d6ca", "#ff7a8a"]
    for i, (label, value) in enumerate(zip(labels, values)):
        bh = value / max_value * plot_h
        bx = left + i * slot + (slot - bar_w) / 2
        by = top + plot_h - bh
        parts.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" rx="7" fill="{palette[i % len(palette)]}"/>')
        parts.append(f'<text x="{bx+bar_w/2:.2f}" y="{by-9:.2f}" text-anchor="middle" fill="#edf2fa" font-family="ui-monospace,monospace" font-size="13" font-weight="700">{value:.3f}</text>')
        wrapped = label.split("|")
        for line_idx, line in enumerate(wrapped):
            parts.append(f'<text x="{bx+bar_w/2:.2f}" y="{height-bottom+26+line_idx*16}" text-anchor="middle" fill="#aeb9ca" font-family="Inter,system-ui,sans-serif" font-size="11">{html.escape(line)}</text>')
    parts.append(f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">{html.escape(unit)}</text>')
    return svg_frame(title, subtitle, "".join(parts), width, height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convergence", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, default=Path("paper/plots"))
    parser.add_argument("--result-dir", type=Path, default=Path("paper/results"))
    args = parser.parse_args()
    convergence = json.loads(args.convergence.read_text())
    weights = json.loads(args.weights.read_text())
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    (args.plot_dir / "h100_dynamic_rht_convergence.svg").write_text(
        convergence_svg(convergence)
    )
    summaries = convergence["summary"]
    (args.plot_dir / "h100_dynamic_rht_training_throughput.svg").write_text(
        bar_svg(
            "Short-run training throughput",
            "Includes FP32 master materialization, backward, gradient mapping, and AdamW",
            ["BF16", "TE block-FP8", "Dynamic RHT|block-FP8"],
            [
                summaries["bf16"]["tokens_per_second"] / 1000,
                summaries["block_fp8"]["tokens_per_second"] / 1000,
                summaries["dynamic_rht_block_fp8"]["tokens_per_second"] / 1000,
            ],
            "Thousands of tokens / second",
        )
    )
    (args.plot_dir / "h100_dynamic_rht_weight_overhead.svg").write_text(
        bar_svg(
            "Dynamic-weight normalized step cost",
            "Each dynamic method is interleaved with static rotation · AdamW kernels excluded",
            ["Static rotated", "BF16 working|dynamic", "Fused 2D FP8|dynamic"],
            [
                1.0,
                weights["dynamic_train_ms"] / weights["static_train_ms"],
                weights["fused_quantized_dynamic_train_ms"]
                / statistics.median(
                    weights["samples_ms"]["static_train_paired_with_fused"]
                ),
            ],
            "Step time relative to static rotation",
        )
    )

    with (args.result_dir / "h100_nvl_dynamic_rht_convergence_curves.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["step", *LABELS])
        for index in range(convergence["steps"]):
            writer.writerow(
                [index + 1]
                + [convergence["curves"][name]["loss"][index] for name in LABELS]
            )
    with (args.result_dir / "h100_nvl_dynamic_rht_training_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["method", "initial_loss", "final_loss", "loss_reduction", "median_step_ms", "tokens_per_second"]
        )
        for name in LABELS:
            row = summaries[name]
            writer.writerow(
                [name, row["initial_loss"], row["final_loss"], row["loss_reduction"], row["median_step_ms"], row["tokens_per_second"]]
            )
    with (args.result_dir / "h100_nvl_dynamic_rht_weight_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["comparison", "left_ms", "right_ms", "right_over_left"])
        static_fused = statistics.median(
            weights["samples_ms"]["static_train_paired_with_fused"]
        )
        writer.writerow(
            [
                "static_vs_dynamic_bf16_working",
                weights["static_train_ms"],
                weights["dynamic_train_ms"],
                weights["dynamic_train_ms"] / weights["static_train_ms"],
            ]
        )
        writer.writerow(
            [
                "static_vs_dynamic_fused_2d_fp8",
                static_fused,
                weights["fused_quantized_dynamic_train_ms"],
                weights["fused_quantized_dynamic_train_ms"] / static_fused,
            ]
        )
        writer.writerow(
            [
                "dynamic_bf16_working_vs_fused_2d_fp8",
                weights["matched_dynamic_train_ms"],
                weights["matched_fused_quantized_dynamic_train_ms"],
                weights["matched_fused_quantized_dynamic_train_ms"]
                / weights["matched_dynamic_train_ms"],
            ]
        )
    print(f"wrote plots to {args.plot_dir} and summaries to {args.result_dir}")


if __name__ == "__main__":
    main()
