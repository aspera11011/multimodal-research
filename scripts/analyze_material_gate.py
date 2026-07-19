#!/usr/bin/env python3
"""Summarize accuracy and illumination sensitivity at the region level."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def bootstrap_ci(values: list[float], seed: int = 20260719, draws: int = 5000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]

    by_region = defaultdict(list)
    by_class = defaultdict(list)
    for row in rows:
        by_region[row["region_id"]].append(row)
        by_class[row["material_label"]].append(float(row["correct"]))

    region_accuracy = []
    region_consistency = []
    any_flip = []
    region_details = []
    for region_id, group in sorted(by_region.items()):
        predictions = [r["predicted_label"] or "<invalid>" for r in group]
        counts = Counter(predictions)
        consistency = max(counts.values()) / len(group)
        accuracy = sum(bool(r["correct"]) for r in group) / len(group)
        region_accuracy.append(accuracy)
        region_consistency.append(consistency)
        any_flip.append(float(len(counts) > 1))
        region_details.append(
            {
                "region_id": region_id,
                "scene": group[0]["scene"],
                "material_label": group[0]["material_label"],
                "accuracy": accuracy,
                "consistency": consistency,
                "any_flip": len(counts) > 1,
                "prediction_counts": dict(counts),
            }
        )

    class_accuracy = {label: round(float(np.mean(values)), 4) for label, values in sorted(by_class.items())}
    macro_accuracy = round(float(np.mean(list(class_accuracy.values()))), 4) if class_accuracy else float("nan")
    summary = {
        "sample_count": len(rows),
        "region_count": len(by_region),
        "scene_count": len({r["scene"] for r in rows}),
        "sample_accuracy": round(float(np.mean([r["correct"] for r in rows])), 4),
        "macro_class_accuracy": macro_accuracy,
        "mean_region_accuracy": round(float(np.mean(region_accuracy)), 4),
        "mean_region_accuracy_ci95": bootstrap_ci(region_accuracy),
        "mean_consistency": round(float(np.mean(region_consistency)), 4),
        "mean_consistency_ci95": bootstrap_ci(region_consistency),
        "regions_with_any_flip_rate": round(float(np.mean(any_flip)), 4),
        "regions_with_any_flip_ci95": bootstrap_ci(any_flip),
        "invalid_output_count": sum(r["predicted_label"] is None for r in rows),
        "class_accuracy": class_accuracy,
        "region_details": region_details,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RGB-only material constancy failure gate",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Regions / scenes: {summary['region_count']} / {summary['scene_count']}",
        f"- Sample accuracy: {summary['sample_accuracy']:.1%}",
        f"- Macro class accuracy: {summary['macro_class_accuracy']:.1%}",
        f"- Mean region consistency: {summary['mean_consistency']:.1%}",
        f"- Regions with any answer flip: {summary['regions_with_any_flip_rate']:.1%}",
        f"- Invalid outputs: {summary['invalid_output_count']}",
        "",
        "## Per-class accuracy",
        "",
    ]
    lines.extend(f"- {label}: {accuracy:.1%}" for label, accuracy in class_accuracy.items())
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
