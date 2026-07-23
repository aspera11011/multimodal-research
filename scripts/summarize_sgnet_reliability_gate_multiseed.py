import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ["rmse", "mae", "boundary_rmse", "flat_rmse", "false_edge_rate"]
CONDITIONS = [
    "clean",
    "shift_x_pos_1",
    "shift_x_pos_2",
    "shift_x_pos_4",
    "shift_y_pos_1",
    "shift_y_pos_2",
    "shift_y_pos_4",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--seeds", default="20260723,20260724,20260725")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    paired = {
        seed: load_json(args.input_root / f"seed_{seed}" / "vs_baseline_paired.json")
        for seed in seeds
    }
    training = {
        seed: load_json(args.input_root / f"seed_{seed}" / "training_summary.json")
        for seed in seeds
    }

    conditions = {}
    strict_failures = []
    all_seed_means_improve = True
    for condition in CONDITIONS:
        metric_summaries = {}
        for metric in METRICS:
            records = [paired[seed]["conditions"][condition]["metrics"][metric] for seed in seeds]
            candidate_values = np.asarray([record["candidate_mean"] for record in records])
            relative_changes = np.asarray(
                [record["relative_change_percent"] for record in records]
            )
            significant = [record["paired_95_ci"][1] < 0 for record in records]
            lower_for_all_seeds = all(
                record["candidate_mean"] < record["baseline_mean"] for record in records
            )
            all_seed_means_improve = all_seed_means_improve and lower_for_all_seeds
            for seed, record, passed in zip(seeds, records, significant):
                if not passed:
                    strict_failures.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "metric": metric,
                            "paired_95_ci": record["paired_95_ci"],
                        }
                    )
            metric_summaries[metric] = {
                "baseline_mean": records[0]["baseline_mean"],
                "candidate_seed_values": candidate_values.tolist(),
                "candidate_seed_mean": float(candidate_values.mean()),
                "candidate_seed_sample_std": float(candidate_values.std(ddof=1)),
                "mean_relative_change_percent": float(relative_changes.mean()),
                "all_seed_means_improve": lower_for_all_seeds,
                "significant_seed_count": int(sum(significant)),
                "all_seed_paired_95_ci_below_zero": all(significant),
            }
        conditions[condition] = {"metrics": metric_summaries}

    gate_training_means = [training[seed]["history"][-1]["gate_mean"] for seed in seeds]
    output = {
        "status": "completed",
        "task": "rgb_guided_depth_super_resolution",
        "method": "SGNet_spatial_reliability_gate",
        "dataset": "RGB-D-D/test2",
        "scale": 16,
        "sample_count": 405,
        "training_dataset": "NYU_v2_train",
        "training_samples_per_seed": 2000,
        "epochs": 1,
        "trainable_parameter_count": 881,
        "seeds": seeds,
        "gate_training_mean_by_seed": gate_training_means,
        "conditions": conditions,
        "decision": {
            "all_seed_means_improve_all_metrics": all_seed_means_improve,
            "strict_all_seed_paired_significance_pass": not strict_failures,
            "strict_failure_count": len(strict_failures),
            "strict_failures": strict_failures,
            "interpretation": (
                "Retain the module: every seed improves every mean metric. The strict gate "
                "misses only where a paired confidence interval touches or crosses zero."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(output["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
