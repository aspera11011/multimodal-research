import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate_sgnet_rgbdd_x16 import (
    aggregate,
    calculate_metrics,
    load_pairs,
    prepare_tensors,
    sha256,
)
from sgnet_reliability_gate import SGNetWithReliabilityGate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgnet-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=16)
    parser.add_argument("--num-feats", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shift-x", type=int, default=0)
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument(
        "--gate-mode",
        choices=("learned", "identity", "shuffled", "constant_mean"),
        default="learned",
    )
    parser.add_argument("--crop-border", type=int, default=6)
    parser.add_argument("--depth-edge-threshold", type=float, default=2.0)
    parser.add_argument("--rgb-edge-threshold", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.sgnet_dir.resolve()))
    from models.SGNet import SGNet

    device = torch.device(args.device)
    base_model = SGNet(num_feats=args.num_feats, kernel_size=3, scale=args.scale)
    base_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    gate_checkpoint = torch.load(args.gate_checkpoint, map_location=device)
    model = SGNetWithReliabilityGate(
        base_model,
        hidden_channels=gate_checkpoint["hidden_channels"],
        initial_bias=gate_checkpoint["initial_bias"],
    )
    model.reliability_gate.load_state_dict(gate_checkpoint["gate_state_dict"])
    model.to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for index, (name, rgb, depth) in enumerate(load_pairs(args.data_root, args.scale)):
            if args.max_samples is not None and index >= args.max_samples:
                break
            guidance, low_resolution, shifted_rgb = prepare_tensors(
                rgb, depth, args.scale, args.shift_x, args.shift_y, device
            )
            prediction, _, gate = model(
                (guidance, low_resolution),
                gate_mode=args.gate_mode,
                return_gate=True,
            )
            prediction = prediction[0, 0].detach().cpu().numpy()
            metrics = calculate_metrics(
                prediction,
                depth,
                shifted_rgb,
                args.crop_border,
                args.depth_edge_threshold,
                args.rgb_edge_threshold,
            )
            records.append(
                {
                    "index": index,
                    "name": name,
                    "gate_mean": float(gate.mean().item()),
                    "gate_std": float(gate.std().item()),
                    "gate_suppressed_fraction": float((gate < 0.9).float().mean().item()),
                    **metrics,
                }
            )

    with (args.output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "completed",
        "task": "rgb_guided_depth_super_resolution",
        "method": "SGNet_spatial_reliability_gate",
        "gate_mode": args.gate_mode,
        "dataset": "RGB-D-D/test2",
        "scale": args.scale,
        "sample_count": len(records),
        "base_checkpoint_sha256": sha256(args.checkpoint),
        "gate_checkpoint_sha256": sha256(args.gate_checkpoint),
        "shift_x": args.shift_x,
        "shift_y": args.shift_y,
        "crop_border": args.crop_border,
        "metrics_unit": "8-bit depth levels",
        "metrics": aggregate(records),
        "gate_statistics": {
            "mean": float(np.mean([record["gate_mean"] for record in records])),
            "std_mean": float(np.mean([record["gate_std"] for record in records])),
            "suppressed_fraction": float(
                np.mean([record["gate_suppressed_fraction"] for record in records])
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
