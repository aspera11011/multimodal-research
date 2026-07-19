#!/usr/bin/env python3
"""Download a small Multi-Illumination subset and build an RGB-only gate manifest."""

from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


BASE_URL = "https://data.csail.mit.edu/multilum"
SCENES_URL = "https://raw.githubusercontent.com/lmurmann/multi_illumination/master/scenes.json"
MATERIALS = {
    4: "ceramic",
    8: "fabric/cloth",
    12: "glass",
    13: "granite/marble",
    15: "leather",
    17: "metal",
    21: "paper/tissue",
    22: "clear plastic",
    23: "opaque plastic",
    27: "tile",
    32: "wood",
    33: "stone",
    35: "carpet/rug",
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def download_mask(scene: str, mip: int, path: Path) -> None:
    if not path.exists():
        path.write_bytes(fetch(f"{BASE_URL}/{scene}/materials_mip{mip}.png"))


def read_mask(path: Path) -> np.ndarray:
    image = Image.open(path)
    if image.mode != "P":
        raise ValueError(f"Expected indexed PNG mask, got {image.mode}: {path}")
    return np.asarray(image, dtype=np.uint8)


def component_candidates(mask: np.ndarray, cfg: dict, scene: str) -> list[dict]:
    records = []
    height, width = mask.shape
    crop_size = int(cfg["crop_size"])
    half = crop_size // 2
    for material_id in cfg["target_material_ids"]:
        labels, count = ndimage.label(mask == material_id)
        for component_id in range(1, count + 1):
            component = labels == component_id
            area = int(component.sum())
            if area < int(cfg["min_component_area"]):
                continue
            cy, cx = np.unravel_index(np.argmax(ndimage.distance_transform_edt(component)), component.shape)
            x0 = max(0, min(width - crop_size, int(cx) - half))
            y0 = max(0, min(height - crop_size, int(cy) - half))
            x1, y1 = x0 + crop_size, y0 + crop_size
            purity = float(component[y0:y1, x0:x1].mean())
            if purity < float(cfg["min_crop_purity"]):
                continue
            records.append(
                {
                    "scene": scene,
                    "material_id": int(material_id),
                    "material_label": MATERIALS[int(material_id)],
                    "component_id": component_id,
                    "component_area": area,
                    "bbox": [x0, y0, x1, y1],
                    "crop_purity": purity,
                }
            )
    return records


def select_scenes(scene_candidates: dict[str, list[dict]], count: int) -> list[str]:
    selected = []
    class_counts: Counter = Counter()
    remaining = set(scene_candidates)
    while remaining and len(selected) < count:
        def score(scene: str) -> tuple:
            classes = {row["material_id"] for row in scene_candidates[scene]}
            novelty = sum(1.0 / (1 + class_counts[c]) for c in classes)
            return novelty, len(classes), len(scene_candidates[scene]), scene

        best = max(remaining, key=score)
        remaining.remove(best)
        selected.append(best)
        class_counts.update(row["material_id"] for row in scene_candidates[best])
    return selected


def select_balanced_regions(
    scene_candidates: dict[str, list[dict]], selected_scenes: list[str], cfg: dict
) -> list[dict]:
    """Round-robin material classes after applying a per-scene/class cap."""
    pools: defaultdict[int, list[dict]] = defaultdict(list)
    cap = int(cfg["max_regions_per_scene_class"])
    for scene in selected_scenes:
        per_class: defaultdict[int, list[dict]] = defaultdict(list)
        for row in scene_candidates[scene]:
            per_class[row["material_id"]].append(row)
        for material_id, rows in per_class.items():
            ordered = sorted(
                rows,
                key=lambda row: (row["crop_purity"], row["component_area"]),
                reverse=True,
            )
            pools[material_id].extend(ordered[:cap])

    for material_id in pools:
        pools[material_id].sort(
            key=lambda row: (row["crop_purity"], row["component_area"], row["scene"]),
            reverse=True,
        )

    selected: list[dict] = []
    target_ids = [int(material_id) for material_id in cfg["target_material_ids"]]
    limit = int(cfg["max_regions"])
    while len(selected) < limit:
        added = False
        for material_id in target_ids:
            if pools[material_id] and len(selected) < limit:
                selected.append(pools[material_id].pop(0))
                added = True
        if not added:
            break
    return selected


def download_scene(scene: str, mip: int, data_root: Path) -> Path:
    scene_dir = data_root / scene
    expected = scene_dir / f"dir_24_mip{mip}.jpg"
    if expected.exists():
        return scene_dir
    archive = fetch(f"{BASE_URL}/{scene}/{scene}_mip{mip}_jpg.zip")
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        handle.extractall(data_root)
    if not expected.exists():
        raise FileNotFoundError(f"Expected image missing after extraction: {expected}")
    return scene_dir


def select_diverse_lights(images: list[np.ndarray], bbox: list[int], count: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    features = []
    for image in images:
        patch = image[y0:y1, x0:x1].astype(np.float32) / 255.0
        features.append(np.concatenate([patch.mean((0, 1)), patch.std((0, 1))]))
    features = np.stack(features)
    luminance = features[:, :3].mean(1)
    selected = list(dict.fromkeys([int(np.argmin(luminance)), int(np.argmax(luminance))]))
    scale = features.std(0, keepdims=True) + 1e-6
    normalized = (features - features.mean(0, keepdims=True)) / scale
    while len(selected) < count:
        distances = np.linalg.norm(normalized[:, None, :] - normalized[selected][None, :, :], axis=2)
        nearest = distances.min(1)
        nearest[selected] = -1
        selected.append(int(np.argmax(nearest)))
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--scenes-json", type=Path)
    args = parser.parse_args()
    cfg = load_json(args.config)
    mip = int(cfg["mip"])
    rng = random.Random(int(cfg["seed"]))

    experiment_id = cfg.get("experiment_id", "material_constancy_rgb_gate_v1")
    raw_root = args.workspace / "data" / "raw" / f"multi_illumination_mip{mip}"
    mask_root = raw_root / "masks"
    crop_root = args.workspace / "data" / "processed" / experiment_id / "crops"
    manifest_root = args.workspace / "experiments" / "manifests" / experiment_id
    for path in [raw_root, mask_root, crop_root, manifest_root]:
        path.mkdir(parents=True, exist_ok=True)

    if args.scenes_json:
        scenes = json.loads(args.scenes_json.read_text(encoding="utf-8"))
    else:
        scenes = json.loads(fetch(SCENES_URL).decode("utf-8"))
    by_room: defaultdict[str, list[str]] = defaultdict(list)
    for scene in scenes:
        if scene["room_type"] in cfg["room_types"] and not scene["name"].startswith("everett"):
            by_room[scene["room_type"]].append(scene["name"])
    candidate_names = []
    for room_type in cfg["room_types"]:
        names = sorted(by_room[room_type])
        rng.shuffle(names)
        candidate_names.extend(names[: int(cfg["candidate_scenes_per_room_type"])])

    scene_candidates: dict[str, list[dict]] = {}
    failures = []
    for scene in candidate_names:
        mask_path = mask_root / f"{scene}_materials_mip{mip}.png"
        try:
            download_mask(scene, mip, mask_path)
            records = component_candidates(read_mask(mask_path), cfg, scene)
            if records:
                scene_candidates[scene] = records
        except Exception as exc:
            failures.append({"scene": scene, "error": repr(exc)})

    selected_scenes = select_scenes(scene_candidates, int(cfg["scene_count"]))
    if len(selected_scenes) < int(cfg["scene_count"]):
        raise RuntimeError(f"Only selected {len(selected_scenes)} usable scenes")

    regions = select_balanced_regions(scene_candidates, selected_scenes, cfg)
    minimum = int(cfg.get("min_regions", 1))
    if len(regions) < minimum:
        raise RuntimeError(f"Only selected {len(regions)} regions; required at least {minimum}")

    scene_images: dict[str, list[np.ndarray]] = {}
    for scene in selected_scenes:
        scene_dir = download_scene(scene, mip, raw_root)
        images = [np.asarray(Image.open(scene_dir / f"dir_{direction}_mip{mip}.jpg").convert("RGB")) for direction in range(25)]
        mask_shape = read_mask(mask_root / f"{scene}_materials_mip{mip}.png").shape
        if any(image.shape[:2] != mask_shape for image in images):
            raise ValueError(f"Image/mask size mismatch for {scene}: mask={mask_shape}")
        scene_images[scene] = images

    labels_used = sorted({row["material_label"] for row in regions})
    samples = []
    for region_index, region in enumerate(regions):
        scene = region["scene"]
        lights = select_diverse_lights(scene_images[scene], region["bbox"], int(cfg["lights_per_region"]))
        x0, y0, x1, y1 = region["bbox"]
        region_id = f"{scene}_r{region_index:03d}_{region['material_id']}"
        for light in lights:
            crop = Image.fromarray(scene_images[scene][light][y0:y1, x0:x1])
            output = crop_root / scene / f"{region_id}_d{light:02d}.jpg"
            output.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output, quality=95)
            patch = np.asarray(crop, dtype=np.float32) / 255.0
            samples.append(
                {
                    "sample_id": f"{region_id}_d{light:02d}",
                    "region_id": region_id,
                    "scene": scene,
                    "light_dir": light,
                    "material_id": region["material_id"],
                    "material_label": region["material_label"],
                    "candidate_labels": labels_used,
                    "bbox": region["bbox"],
                    "component_area": region["component_area"],
                    "crop_purity": round(region["crop_purity"], 6),
                    "crop_path": str(output),
                    "mean_rgb": patch.mean((0, 1)).round(6).tolist(),
                    "std_rgb": patch.std((0, 1)).round(6).tolist(),
                }
            )

    manifest_path = manifest_root / "material_constancy_rgb_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "config": cfg,
        "candidate_scene_count": len(candidate_names),
        "usable_candidate_scene_count": len(scene_candidates),
        "selected_scenes": selected_scenes,
        "region_count": len(regions),
        "sample_count": len(samples),
        "candidate_labels": labels_used,
        "regions_per_class": dict(Counter(row["material_label"] for row in regions)),
        "samples_per_class": dict(Counter(row["material_label"] for row in samples)),
        "failed_mask_downloads": failures,
        "manifest": str(manifest_path),
    }
    (manifest_root / "material_constancy_rgb_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
