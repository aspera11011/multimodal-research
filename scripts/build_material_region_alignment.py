#!/usr/bin/env python3
"""Build paired full-scene RGB/albedo inputs for the region-alignment diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def draw_target_box(image: Image.Image, bbox: list[int]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    x0, y0, x1, y1 = map(int, bbox)
    width = max(3, min(output.size) // 160)
    # A white halo keeps the same red marker visible on dark and bright surfaces.
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="white", width=width + 4)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(235, 35, 45), width=width)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mip", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    raw_root = args.workspace / "data" / "raw" / f"multi_illumination_mip{args.mip}"
    out_root = args.workspace / "data" / "processed" / "material_constancy_region_alignment_v1"
    rgb_root = out_root / "rgb_marked"
    albedo_root = out_root / "albedo_marked"
    output_rows = []

    for index, row in enumerate(rows, start=1):
        scene = row["scene"]
        sample_id = row["sample_id"]
        rgb_full = raw_root / scene / f"dir_{int(row['light_dir'])}_mip{args.mip}.jpg"
        albedo_full = Path(row["albedo_full_path"])
        if not rgb_full.is_file() or not albedo_full.is_file():
            raise FileNotFoundError(f"Missing full image for {sample_id}: {rgb_full} / {albedo_full}")

        rgb_image = Image.open(rgb_full).convert("RGB")
        albedo_image = Image.open(albedo_full).convert("RGB")
        if rgb_image.size != albedo_image.size:
            raise ValueError(f"Geometry mismatch for {sample_id}: {rgb_image.size} vs {albedo_image.size}")

        rgb_marked = rgb_root / scene / f"{sample_id}_rgb_marked.jpg"
        albedo_marked = albedo_root / scene / f"{sample_id}_albedo_marked.png"
        rgb_marked.parent.mkdir(parents=True, exist_ok=True)
        albedo_marked.parent.mkdir(parents=True, exist_ok=True)
        draw_target_box(rgb_image, row["bbox"]).save(rgb_marked, quality=95)
        draw_target_box(albedo_image, row["bbox"]).save(albedo_marked)

        output_rows.append(
            {
                **row,
                "rgb_full_path": str(rgb_full),
                "rgb_marked_path": str(rgb_marked),
                "albedo_marked_path": str(albedo_marked),
                "marker": {
                    "type": "matched_rectangle",
                    "color": "red_with_white_halo",
                    "bbox": row["bbox"],
                },
            }
        )
        print(f"[{index}/{len(rows)}] {sample_id}", flush=True)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"sample_count": len(output_rows), "output_manifest": str(args.output_manifest)}))


if __name__ == "__main__":
    main()
