#!/usr/bin/env python3
"""Export safetensors weights into visualization-friendly CSV/JSON files.

Outputs:
1) layer_stats.csv          - Per-tensor statistics.
2) global_histogram.csv     - Histogram over all parameter values.
3) per_layer_hist.json      - Histogram per tensor.
4) summary.json             - Global summary and top-k tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open


def compute_hist(values: np.ndarray, bins: int) -> dict:
    if values.size == 0:
        return {
            "bin_edges": [0.0, 1.0],
            "counts": [0],
            "min": 0.0,
            "max": 0.0,
        }

    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        eps = abs(vmin) * 1e-6 + 1e-6
        vmin -= eps
        vmax += eps

    counts, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "min": vmin,
        "max": vmax,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export visualization data from a .safetensors model")
    parser.add_argument("--input", required=True, help="Path to model.safetensors")
    parser.add_argument("--output-dir", required=True, help="Directory for exported files")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bins (default: 80)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_rows: list[dict] = []
    per_layer_hist: dict[str, dict] = {}

    global_values = []
    total_params = 0

    with safe_open(str(input_path), framework="np") as sf:
        keys = list(sf.keys())
        for name in keys:
            tensor = sf.get_tensor(name)
            arr = np.asarray(tensor, dtype=np.float64)
            flat = arr.reshape(-1)

            numel = int(flat.size)
            if numel == 0:
                stats = {
                    "name": name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "numel": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "l2_norm": 0.0,
                    "abs_mean": 0.0,
                    "p01": 0.0,
                    "p50": 0.0,
                    "p99": 0.0,
                    "zero_ratio": 0.0,
                }
            else:
                mean = float(flat.mean())
                std = float(flat.std())
                vmin = float(flat.min())
                vmax = float(flat.max())
                l2_norm = float(np.linalg.norm(flat))
                abs_mean = float(np.abs(flat).mean())
                p01, p50, p99 = [float(x) for x in np.percentile(flat, [1, 50, 99])]
                zero_ratio = float(np.mean(flat == 0.0))

                stats = {
                    "name": name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "numel": numel,
                    "mean": mean,
                    "std": std,
                    "min": vmin,
                    "max": vmax,
                    "l2_norm": l2_norm,
                    "abs_mean": abs_mean,
                    "p01": p01,
                    "p50": p50,
                    "p99": p99,
                    "zero_ratio": zero_ratio,
                }

                global_values.append(flat)
                per_layer_hist[name] = compute_hist(flat, bins=args.bins)

            total_params += numel
            layer_rows.append(stats)

    if global_values:
        concat = np.concatenate(global_values)
    else:
        concat = np.array([], dtype=np.float64)

    global_hist = compute_hist(concat, bins=args.bins)

    summary = {
        "input": str(input_path),
        "num_tensors": len(layer_rows),
        "total_params": total_params,
        "global": {
            "mean": float(concat.mean()) if concat.size else 0.0,
            "std": float(concat.std()) if concat.size else 0.0,
            "min": float(concat.min()) if concat.size else 0.0,
            "max": float(concat.max()) if concat.size else 0.0,
            "abs_mean": float(np.abs(concat).mean()) if concat.size else 0.0,
            "zero_ratio": float(np.mean(concat == 0.0)) if concat.size else 0.0,
        },
        "top_tensors_by_numel": sorted(
            [{"name": r["name"], "numel": r["numel"]} for r in layer_rows],
            key=lambda x: x["numel"],
            reverse=True,
        )[:30],
        "top_tensors_by_std": sorted(
            [{"name": r["name"], "std": r["std"]} for r in layer_rows],
            key=lambda x: x["std"],
            reverse=True,
        )[:30],
    }

    layer_csv = output_dir / "layer_stats.csv"
    with layer_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "shape",
            "dtype",
            "numel",
            "mean",
            "std",
            "min",
            "max",
            "l2_norm",
            "abs_mean",
            "p01",
            "p50",
            "p99",
            "zero_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in layer_rows:
            out = row.copy()
            out["shape"] = json.dumps(out["shape"], ensure_ascii=False)
            writer.writerow(out)

    global_hist_csv = output_dir / "global_histogram.csv"
    with global_hist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_left", "bin_right", "count"])
        edges = global_hist["bin_edges"]
        counts = global_hist["counts"]
        for i, c in enumerate(counts):
            writer.writerow([edges[i], edges[i + 1], c])

    with (output_dir / "per_layer_hist.json").open("w", encoding="utf-8") as f:
        json.dump(per_layer_hist, f, ensure_ascii=False)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Exported files to: {output_dir}")
    print(f"- {layer_csv.name}")
    print(f"- {global_hist_csv.name}")
    print("- per_layer_hist.json")
    print("- summary.json")


if __name__ == "__main__":
    main()
