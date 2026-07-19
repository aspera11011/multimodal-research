#!/usr/bin/env python3
"""Extract per-image roughness/metallicity evidence from full-frame Marigold IID predictions."""

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
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.input_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: defaultdict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], int(row["light_dir"]))].append(row)
    keys = sorted(grouped)

    pipe = MarigoldIntrinsicsPipeline.from_pretrained(
        cfg["model_local_dir"],
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    raw_root = args.workspace / "data" / "raw" / "multi_illumination_mip4"
    output_rows = []
    for index, (scene, light_dir) in enumerate(keys, start=1):
        image = Image.open(raw_root / scene / f"dir_{light_dir}_mip4.jpg").convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(int(cfg["seed"]) + index)
        with torch.inference_mode():
            result = pipe(
                image,
                num_inference_steps=int(cfg["num_inference_steps"]),
                ensemble_size=int(cfg["ensemble_size"]),
                processing_resolution=int(cfg["processing_resolution"]),
                match_input_resolution=True,
                generator=generator,
                output_type="pt",
            )
        material = result.prediction[1].float().cpu()
        roughness, metallicity = material[0], material[1]
        for row in grouped[(scene, light_dir)]:
            x0, y0, x1, y1 = row["bbox"]
            roughness_mean = float(roughness[y0:y1, x0:x1].mean())
            metallicity_mean = float(metallicity[y0:y1, x0:x1].mean())
            evidence = (
                "A noisy intrinsic estimator reports surface roughness "
                f"{roughness_mean:.2f} and metallicity {metallicity_mean:.2f} on 0-to-1 scales."
            )
            output_rows.append(
                {
                    **row,
                    "roughness_mean": round(roughness_mean, 6),
                    "metallicity_mean": round(metallicity_mean, 6),
                    "intrinsic_evidence": evidence,
                }
            )
        print(f"[{index}/{len(keys)}] {scene} light={light_dir}", flush=True)

    order = {row["sample_id"]: idx for idx, row in enumerate(rows)}
    output_rows.sort(key=lambda row: order[row["sample_id"]])
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"sample_count": len(output_rows), "scene_light_count": len(keys)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
