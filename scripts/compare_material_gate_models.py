#!/usr/bin/env python3
"""Compare region-level illumination flips across two material-classification VLMs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def flip_map(rows: dict[str, dict]) -> dict[str, bool]:
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows.values():
        grouped[row["region_id"]].add(row["predicted_label"] or "<invalid>")
    return {region_id: len(labels) > 1 for region_id, labels in grouped.items()}


def bootstrap_ci(values: list[float], seed: int = 20260719, draws: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--a-name", required=True)
    parser.add_argument("--b-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    a_rows, b_rows = read_rows(args.a), read_rows(args.b)
    if set(a_rows) != set(b_rows):
        raise ValueError("Prediction files do not contain identical sample IDs")
    for sample_id in a_rows:
        for key in ["region_id", "scene", "material_label", "light_dir"]:
            if a_rows[sample_id][key] != b_rows[sample_id][key]:
                raise ValueError(f"Mismatched {key} for {sample_id}")

    a_flips, b_flips = flip_map(a_rows), flip_map(b_rows)
    if set(a_flips) != set(b_flips):
        raise ValueError("Prediction files do not contain identical region IDs")
    region_ids = sorted(a_flips)
    both = [float(a_flips[r] and b_flips[r]) for r in region_ids]
    either = [float(a_flips[r] or b_flips[r]) for r in region_ids]
    a_only = [float(a_flips[r] and not b_flips[r]) for r in region_ids]
    b_only = [float(b_flips[r] and not a_flips[r]) for r in region_ids]
    neither = [float(not a_flips[r] and not b_flips[r]) for r in region_ids]
    sample_agreement = np.mean(
        [a_rows[s]["predicted_label"] == b_rows[s]["predicted_label"] for s in sorted(a_rows)]
    )
    intersection = int(sum(both))
    union = int(sum(either))
    summary = {
        "model_a": args.a_name,
        "model_b": args.b_name,
        "sample_count": len(a_rows),
        "region_count": len(region_ids),
        "sample_prediction_agreement": round(float(sample_agreement), 4),
        "both_flip_count": intersection,
        "both_flip_rate": round(float(np.mean(both)), 4),
        "both_flip_ci95": bootstrap_ci(both),
        "either_flip_count": union,
        "either_flip_rate": round(float(np.mean(either)), 4),
        "either_flip_ci95": bootstrap_ci(either),
        "a_only_flip_count": int(sum(a_only)),
        "b_only_flip_count": int(sum(b_only)),
        "neither_flip_count": int(sum(neither)),
        "flip_set_jaccard": round(intersection / union, 4) if union else 1.0,
        "both_flip_region_ids": [r for r in region_ids if a_flips[r] and b_flips[r]],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Cross-model material constancy comparison",
        "",
        f"- Models: {args.a_name} vs {args.b_name}",
        f"- Samples / regions: {summary['sample_count']} / {summary['region_count']}",
        f"- Sample prediction agreement: {summary['sample_prediction_agreement']:.1%}",
        f"- Both models flip: {summary['both_flip_count']}/{summary['region_count']} "
        f"({summary['both_flip_rate']:.1%}, CI {summary['both_flip_ci95']})",
        f"- Either model flips: {summary['either_flip_count']}/{summary['region_count']} "
        f"({summary['either_flip_rate']:.1%}, CI {summary['either_flip_ci95']})",
        f"- A only / B only / neither: {summary['a_only_flip_count']} / "
        f"{summary['b_only_flip_count']} / {summary['neither_flip_count']}",
        f"- Flip-set Jaccard: {summary['flip_set_jaccard']:.3f}",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
