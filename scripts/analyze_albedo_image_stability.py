#!/usr/bin/env python3
"""Measure paired cross-light pixel/color variation for RGB and estimated albedo crops."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def ci(values: np.ndarray, seed: int = 20260719, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return [round(float(np.quantile(means, 0.025)), 6), round(float(np.quantile(means, 0.975)), 6)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["region_id"]].append(row)
    details = []
    for region_id, group in sorted(grouped.items()):
        rgb = np.stack([np.asarray(Image.open(r["rgb_crop_path"]).convert("RGB"), dtype=np.float32) / 255 for r in group])
        albedo = np.stack(
            [np.asarray(Image.open(r["albedo_crop_path"]).convert("RGB"), dtype=np.float32) / 255 for r in group]
        )
        rgb_pixel_std = float(rgb.std(axis=0).mean())
        albedo_pixel_std = float(albedo.std(axis=0).mean())
        rgb_mean_color_std = float(rgb.mean(axis=(1, 2)).std(axis=0).mean())
        albedo_mean_color_std = float(albedo.mean(axis=(1, 2)).std(axis=0).mean())
        details.append(
            {
                "region_id": region_id,
                "material_label": group[0]["material_label"],
                "rgb_pixel_std": rgb_pixel_std,
                "albedo_pixel_std": albedo_pixel_std,
                "pixel_std_delta": albedo_pixel_std - rgb_pixel_std,
                "rgb_mean_color_std": rgb_mean_color_std,
                "albedo_mean_color_std": albedo_mean_color_std,
                "mean_color_std_delta": albedo_mean_color_std - rgb_mean_color_std,
            }
        )
    pixel_delta = np.asarray([d["pixel_std_delta"] for d in details])
    color_delta = np.asarray([d["mean_color_std_delta"] for d in details])
    summary = {
        "region_count": len(details),
        "rgb_mean_pixel_std": round(float(np.mean([d["rgb_pixel_std"] for d in details])), 6),
        "albedo_mean_pixel_std": round(float(np.mean([d["albedo_pixel_std"] for d in details])), 6),
        "pixel_std_delta": round(float(pixel_delta.mean()), 6),
        "pixel_std_delta_ci95": ci(pixel_delta),
        "regions_pixel_variation_reduced": int(np.sum(pixel_delta < 0)),
        "rgb_mean_color_std": round(float(np.mean([d["rgb_mean_color_std"] for d in details])), 6),
        "albedo_mean_color_std": round(float(np.mean([d["albedo_mean_color_std"] for d in details])), 6),
        "mean_color_std_delta": round(float(color_delta.mean()), 6),
        "mean_color_std_delta_ci95": ci(color_delta),
        "regions_mean_color_variation_reduced": int(np.sum(color_delta < 0)),
        "region_details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "region_details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
