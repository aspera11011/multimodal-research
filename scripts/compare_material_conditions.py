#!/usr/bin/env python3
"""Paired region-bootstrap comparison of an intervention against RGB baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return {row["sample_id"]: row for row in rows}


def region_metrics(rows: dict[str, dict]) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows.values():
        grouped[row["region_id"]].append(row)
    output = {}
    for region_id, group in grouped.items():
        predictions = [row["predicted_label"] or "<invalid>" for row in group]
        output[region_id] = {
            "accuracy": float(np.mean([row["correct"] for row in group])),
            "consistency": max(Counter(predictions).values()) / len(predictions),
            "flip": float(len(set(predictions)) > 1),
        }
    return output


def paired_ci(values: np.ndarray, seed: int = 20260719, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--intervention", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline, intervention = read_rows(args.baseline), read_rows(args.intervention)
    if set(baseline) != set(intervention):
        raise ValueError("Baseline and intervention sample IDs differ")
    for sample_id in baseline:
        for key in ["region_id", "scene", "light_dir", "material_label"]:
            if baseline[sample_id][key] != intervention[sample_id][key]:
                raise ValueError(f"Mismatched {key} for {sample_id}")

    base_region, int_region = region_metrics(baseline), region_metrics(intervention)
    region_ids = sorted(base_region)
    deltas = {
        metric: np.asarray([int_region[r][metric] - base_region[r][metric] for r in region_ids], dtype=float)
        for metric in ["accuracy", "consistency", "flip"]
    }
    summary = {
        "model": args.model_name,
        "condition": args.condition_name,
        "sample_count": len(baseline),
        "region_count": len(region_ids),
        "baseline_sample_accuracy": round(float(np.mean([row["correct"] for row in baseline.values()])), 4),
        "intervention_sample_accuracy": round(
            float(np.mean([row["correct"] for row in intervention.values()])), 4
        ),
        "accuracy_delta": round(float(deltas["accuracy"].mean()), 4),
        "accuracy_delta_ci95": paired_ci(deltas["accuracy"]),
        "consistency_delta": round(float(deltas["consistency"].mean()), 4),
        "consistency_delta_ci95": paired_ci(deltas["consistency"]),
        "flip_rate_delta": round(float(deltas["flip"].mean()), 4),
        "flip_rate_delta_ci95": paired_ci(deltas["flip"]),
        "regions_accuracy_improved": int(np.sum(deltas["accuracy"] > 0)),
        "regions_accuracy_worsened": int(np.sum(deltas["accuracy"] < 0)),
        "regions_flip_resolved": int(np.sum(deltas["flip"] < 0)),
        "regions_newly_flipped": int(np.sum(deltas["flip"] > 0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
