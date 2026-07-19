#!/usr/bin/env python3
"""Generate full-frame Marigold IID albedo maps, then crop using a fixed manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import MarigoldIntrinsicsPipeline
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--limit-images", type=int)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source = Path(cfg["source_manifest"])
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: defaultdict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], int(row["light_dir"]))].append(row)
    keys = sorted(grouped)
    if args.limit_images is not None:
        keys = keys[: args.limit_images]

    pipe = MarigoldIntrinsicsPipeline.from_pretrained(
        cfg["model_local_dir"],
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    mip = 4
    raw_root = args.workspace / "data" / "raw" / f"multi_illumination_mip{mip}"
    output_root = args.workspace / "data" / "processed" / cfg["experiment_id"]
    full_root = output_root / "full_albedo"
    crop_root = output_root / "crops"
    output_rows = []
    for index, (scene, light_dir) in enumerate(keys, start=1):
        input_path = raw_root / scene / f"dir_{light_dir}_mip{mip}.jpg"
        full_path = full_root / scene / f"dir_{light_dir}_mip{mip}_albedo.png"
        if full_path.exists():
            albedo = Image.open(full_path).convert("RGB")
        else:
            image = Image.open(input_path).convert("RGB")
            generator = torch.Generator(device="cuda").manual_seed(int(cfg["seed"]) + index)
            with torch.inference_mode():
                result = pipe(
                    image,
                    num_inference_steps=int(cfg["num_inference_steps"]),
                    ensemble_size=int(cfg["ensemble_size"]),
                    processing_resolution=int(cfg["processing_resolution"]),
                    match_input_resolution=bool(cfg["match_input_resolution"]),
                    generator=generator,
                    output_type="pt",
                )
            albedo = pipe.image_processor.visualize_intrinsics(
                result.prediction, pipe.target_properties
            )[0]["albedo"]
            if albedo.size != image.size:
                raise ValueError(f"Geometry mismatch for {input_path}: {albedo.size} vs {image.size}")
            full_path.parent.mkdir(parents=True, exist_ok=True)
            albedo.save(full_path)

        for row in grouped[(scene, light_dir)]:
            x0, y0, x1, y1 = row["bbox"]
            crop_path = crop_root / scene / f"{row['sample_id']}_albedo.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            albedo.crop((x0, y0, x1, y1)).save(crop_path)
            output_rows.append(
                {
                    **row,
                    "rgb_crop_path": row["crop_path"],
                    "albedo_crop_path": str(crop_path),
                    "albedo_full_path": str(full_path),
                    "albedo_model_id": cfg["model_id"],
                    "albedo_model_revision": cfg["model_revision"],
                    "albedo_seed": int(cfg["seed"]) + index,
                }
            )
        print(f"[{index}/{len(keys)}] {scene} light={light_dir} regions={len(grouped[(scene, light_dir)])}", flush=True)

    order = {row["sample_id"]: idx for idx, row in enumerate(rows)}
    output_rows.sort(key=lambda row: order[row["sample_id"]])
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "unique_scene_light_count": len(keys),
                "sample_count": len(output_rows),
                "output_manifest": str(args.output_manifest),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
