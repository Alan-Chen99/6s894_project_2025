#!/usr/bin/env python3
"""Generate SVG and CSV artifacts for the fused inverse-RHT AdamW result."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


COLORS = {
    "dynamic_rht_block_fp8": "#70a5ff",
    "dynamic_rht_block_fp8_fused_adamw": "#51d6ca",
}
LABELS = {
    "dynamic_rht_block_fp8": "Mapped RHT + PyTorch AdamW",
    "dynamic_rht_block_fp8_fused_adamw": "Fused inverse-RHT + AdamW",
}


def frame(title: str, subtitle: str, body: str, *, height: int = 520) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}" role="img">
<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>
<rect width="100%" height="100%" rx="16" fill="#0f151f"/>
<text x="54" y="42" fill="#edf2fa" font-family="Inter,system-ui,sans-serif" font-size="20" font-weight="700">{html.escape(title)}</text>
<text x="54" y="65" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">{html.escape(subtitle)}</text>
{body}</svg>\n'''


def convergence_svg(payload: dict) -> str:
    width, height = 900, 520
    left, right, top, bottom = 72, 28, 100, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    steps = payload["steps"]
    curves = payload["curves"]
    values = [value for record in curves.values() for value in record["loss"]]
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
        py = y(10**power)
        parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#273246" stroke-dasharray="3 5"/>')
        parts.append(f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" fill="#8e9bb0" font-family="ui-monospace,monospace" font-size="11">10^{power}</text>')
        power += 1
    ticks = sorted({1, steps, *range(20, steps + 1, 20)})
    for step in ticks:
        px = x(step - 1)
        parts.append(f'<text x="{px:.2f}" y="{height-25}" text-anchor="middle" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="11">{step}</text>')
    parts.append(f'<text x="{width/2}" y="{height-7}" text-anchor="middle" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">Optimizer step</text>')
    parts.append(f'<text transform="translate(17 {top+plot_h/2}) rotate(-90)" text-anchor="middle" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">MSE loss (log scale)</text>')
    for index, name in enumerate(LABELS):
        points = " ".join(
            f"{x(i):.2f},{y(value):.2f}"
            for i, value in enumerate(curves[name]["loss"])
        )
        color = COLORS[name]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"/>')
        lx = 390 + index * 250
        parts.append(f'<line x1="{lx}" y1="84" x2="{lx+22}" y2="84" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+28}" y="88" fill="#cbd5e5" font-family="Inter,system-ui,sans-serif" font-size="11">{html.escape(LABELS[name])}</text>')
    return frame(
        "Fused RHT AdamW convergence",
        "Matched 120-step teacher–student SwiGLU regression · H100 NVL",
        "".join(parts),
    )


def performance_svg(payload: dict) -> str:
    width, height = 900, 500
    left, right, top, bottom = 80, 35, 100, 90
    values = [
        payload["reference_step_ms"],
        payload["fused_step_ms"],
        payload["reference_optimizer_tail_ms"],
        payload["fused_optimizer_tail_ms"],
    ]
    labels = ["Mapped|full step", "Fused|full step", "Mapped|optimizer tail", "Fused|optimizer tail"]
    max_value = max(values) * 1.18
    plot_h = height - top - bottom
    slot = (width - left - right) / len(values)
    bar_w = 105
    parts = []
    for tick in range(5):
        value = max_value * tick / 4
        py = top + plot_h * (1 - tick / 4)
        parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#273246" stroke-dasharray="3 5"/>')
        parts.append(f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" fill="#8e9bb0" font-family="ui-monospace,monospace" font-size="11">{value:.1f}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        bx = left + index * slot + (slot - bar_w) / 2
        bh = value / max_value * plot_h
        by = top + plot_h - bh
        color = "#70a5ff" if index % 2 == 0 else "#51d6ca"
        parts.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w}" height="{bh:.2f}" rx="7" fill="{color}"/>')
        parts.append(f'<text x="{bx+bar_w/2:.2f}" y="{by-9:.2f}" text-anchor="middle" fill="#edf2fa" font-family="ui-monospace,monospace" font-size="13" font-weight="700">{value:.3f} ms</text>')
        for line_index, line in enumerate(label.split("|")):
            parts.append(f'<text x="{bx+bar_w/2:.2f}" y="{height-bottom+27+line_index*16}" text-anchor="middle" fill="#aeb9ca" font-family="Inter,system-ui,sans-serif" font-size="11">{html.escape(line)}</text>')
    parts.append(f'<text x="260" y="88" fill="#cbd5e5" font-family="Inter,system-ui,sans-serif" font-size="12">Full-step speedup: {payload["full_step_speedup"]:.2f}×</text>')
    parts.append(f'<text x="600" y="88" fill="#cbd5e5" font-family="Inter,system-ui,sans-serif" font-size="12">Optimizer-tail speedup: {payload["optimizer_tail_speedup"]:.2f}×</text>')
    parts.append(f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">Median latency (ms; lower is better)</text>')
    return frame(
        "Inverse-RHT + AdamW fusion",
        f"4096 tokens · 4096→22016 SwiGLU→4096 · median of {payload.get('repetitions', 1)} processes · H100 NVL",
        "".join(parts),
        height=height,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convergence", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, nargs="+", required=True)
    parser.add_argument("--plot-dir", type=Path, default=Path("paper/plots"))
    parser.add_argument("--result-dir", type=Path, default=Path("paper/results"))
    args = parser.parse_args()
    convergence = json.loads(args.convergence.read_text())
    benchmark_payloads = [json.loads(path.read_text()) for path in args.benchmark]
    scalar_keys = (
        "reference_step_ms",
        "fused_step_ms",
        "full_step_speedup",
        "reference_optimizer_tail_ms",
        "fused_optimizer_tail_ms",
        "optimizer_tail_speedup",
    )
    benchmark = {
        key: statistics.median(payload[key] for payload in benchmark_payloads)
        for key in scalar_keys
    }
    benchmark["repetitions"] = len(benchmark_payloads)
    for phase in ("forward", "backward"):
        if phase in benchmark_payloads[0]:
            benchmark[phase] = {
                key: statistics.median(payload[phase][key] for payload in benchmark_payloads)
                for key in benchmark_payloads[0][phase]
            }
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    (args.plot_dir / "h100_fused_rht_adamw_convergence.svg").write_text(
        convergence_svg(convergence)
    )
    (args.plot_dir / "h100_fused_rht_adamw_performance.svg").write_text(
        performance_svg(benchmark)
    )

    with (args.result_dir / "h100_nvl_fused_rht_adamw_convergence.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["step", *LABELS])
        for index in range(convergence["steps"]):
            writer.writerow(
                [index + 1]
                + [convergence["curves"][name]["loss"][index] for name in LABELS]
            )

    with (args.result_dir / "h100_nvl_fused_rht_adamw_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "mapped", "fused", "speedup_or_ratio"])
        writer.writerow([
            "full_step_ms",
            benchmark["reference_step_ms"],
            benchmark["fused_step_ms"],
            benchmark["full_step_speedup"],
        ])
        writer.writerow([
            "optimizer_tail_ms",
            benchmark["reference_optimizer_tail_ms"],
            benchmark["fused_optimizer_tail_ms"],
            benchmark["optimizer_tail_speedup"],
        ])
        for phase in ("forward", "backward"):
            if phase in benchmark:
                writer.writerow([
                    f"{phase}_ms",
                    benchmark[phase]["mapped_rht_ms"],
                    benchmark[phase]["fused_adamw_rht_ms"],
                    benchmark[phase]["speedup"],
                ])
        mapped = convergence["summary"]["dynamic_rht_block_fp8"]
        fused = convergence["summary"]["dynamic_rht_block_fp8_fused_adamw"]
        writer.writerow([
            "controlled_median_step_ms",
            mapped["median_step_ms"],
            fused["median_step_ms"],
            mapped["median_step_ms"] / fused["median_step_ms"],
        ])
        writer.writerow([
            "controlled_final_loss",
            mapped["final_loss"],
            fused["final_loss"],
            fused["final_loss"] / mapped["final_loss"],
        ])
    print(f"wrote fused optimizer plots to {args.plot_dir} and CSVs to {args.result_dir}")


if __name__ == "__main__":
    main()
