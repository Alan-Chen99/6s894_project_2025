#!/usr/bin/env python3
"""Aggregate independent direct-TE comparisons into an SVG and CSV."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from pathlib import Path


PHASES = [
    ("forward", "Forward"),
    ("backward", "Backward"),
    ("forward_backward", "Forward + backward"),
    ("full_step", "Complete step"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--plot", type=Path, default=Path("paper/plots/h100_te_vs_fused_rht_adamw.svg")
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("paper/results/h100_nvl_te_vs_fused_rht_adamw_summary.csv"),
    )
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    rows = []
    for key, label in PHASES:
        plain = [payload[key]["plain_te_ms"] for payload in payloads]
        fused = [payload[key]["fused_rht_ms"] for payload in payloads]
        speedups = [payload[key]["speedup"] for payload in payloads]
        rows.append({
            "phase": key,
            "label": label,
            "plain_te_median_ms": statistics.median(plain),
            "fused_rht_median_ms": statistics.median(fused),
            "speedup_median": statistics.median(speedups),
            "speedup_min": min(speedups),
            "speedup_max": max(speedups),
        })

    width, height = 900, 500
    left, right, top, bottom = 90, 35, 100, 92
    plot_h = height - top - bottom
    low, high = 0.75, 1.4
    slot = (width - left - right) / len(rows)
    bar_w = 112

    def y(value: float) -> float:
        return top + (high - value) * plot_h / (high - low)

    parts = []
    for value in (0.8, 1.0, 1.2, 1.4):
        py = y(value)
        stroke = "#73849d" if value == 1.0 else "#273246"
        dash = "5 4" if value == 1.0 else "3 5"
        parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="{stroke}" stroke-dasharray="{dash}"/>')
        parts.append(f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" fill="#8e9bb0" font-family="ui-monospace,monospace" font-size="11">{value:.1f}×</text>')
    base_y = y(low)
    for index, row in enumerate(rows):
        value = row["speedup_median"]
        bx = left + index * slot + (slot - bar_w) / 2
        top_y = y(value)
        one_y = y(1.0)
        by, bh = min(top_y, one_y), abs(top_y - one_y)
        color = "#51d6ca" if value >= 1.0 else "#ffc46b"
        parts.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w}" height="{max(bh, 2):.2f}" rx="7" fill="{color}"/>')
        center = bx + bar_w / 2
        parts.append(f'<line x1="{center:.2f}" y1="{y(row["speedup_min"]):.2f}" x2="{center:.2f}" y2="{y(row["speedup_max"]):.2f}" stroke="#edf2fa" stroke-width="2"/>')
        parts.append(f'<text x="{center:.2f}" y="{top_y-10:.2f}" text-anchor="middle" fill="#edf2fa" font-family="ui-monospace,monospace" font-size="13" font-weight="700">{value:.3f}×</text>')
        parts.append(f'<text x="{center:.2f}" y="{base_y+28:.2f}" text-anchor="middle" fill="#aeb9ca" font-family="Inter,system-ui,sans-serif" font-size="11">{html.escape(row["label"])}</text>')
    body = "".join(parts)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>Final RHT training path versus Transformer Engine</title>
<desc>Median of three independent H100 NVL processes; higher than one favors fused RHT</desc>
<rect width="100%" height="100%" rx="16" fill="#0f151f"/>
<text x="54" y="42" fill="#edf2fa" font-family="Inter,system-ui,sans-serif" font-size="20" font-weight="700">Final path vs ordinary TE block-FP8</text>
<text x="54" y="65" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">4096 tokens · 4096→22016 SwiGLU→4096 · median and range of 3 processes · H100 NVL</text>
{body}
<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" fill="#8e9bb0" font-family="Inter,system-ui,sans-serif" font-size="12">Speedup over plain TE (higher is better)</text>
</svg>\n'''
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.plot.write_text(svg)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "phase",
            "plain_te_median_ms",
            "fused_rht_median_ms",
            "speedup_median",
            "speedup_min",
            "speedup_max",
        ], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "label"})
    print(json.dumps(rows, indent=2))
    print(f"wrote {args.plot} and {args.csv}")


if __name__ == "__main__":
    main()
