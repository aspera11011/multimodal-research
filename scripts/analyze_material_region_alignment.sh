#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/xjy/multimodal-research"
PY="/usr/bin/python3"
export PYTHONPATH="$ROOT/models/python_pkgs${PYTHONPATH:+:$PYTHONPATH}"
R="$ROOT/results/quantitative/material_constancy_region_alignment_v1"
RGB="$ROOT/results/quantitative/material_constancy_rgb_gate_v2"
ALB="$ROOT/results/quantitative/material_constancy_albedo_v1"

summarize() {
  local stem="$1"
  "$PY" "$ROOT/scripts/analyze_material_gate.py" \
    --predictions "$R/${stem}_predictions.jsonl" \
    --summary-json "$R/${stem}_summary.json" \
    --summary-md "$R/${stem}_summary.md" >/dev/null
}

compare() {
  local baseline="$1" intervention="$2" model="$3" condition="$4" output="$5"
  "$PY" "$ROOT/scripts/compare_material_conditions.py" \
    --baseline "$baseline" \
    --intervention "$intervention" \
    --model-name "$model" \
    --condition-name "$condition" \
    --output "$R/$output" >/dev/null
}

for stem in \
  qwen_rgb_marked_albedo_unmarked \
  qwen_rgb_marked_albedo_marked \
  internvl_rgb_marked_albedo_unmarked \
  internvl_rgb_marked_albedo_marked; do
  summarize "$stem"
done

compare "$R/qwen_rgb_marked_albedo_unmarked_predictions.jsonl" "$R/qwen_rgb_marked_albedo_marked_predictions.jsonl" Qwen3-VL-2B shared_marker_vs_single_marker qwen_shared_vs_single.json
compare "$R/internvl_rgb_marked_albedo_unmarked_predictions.jsonl" "$R/internvl_rgb_marked_albedo_marked_predictions.jsonl" InternVL3.5-2B shared_marker_vs_single_marker internvl_shared_vs_single.json

compare "$RGB/qwen3vl2b_predictions_corrected.jsonl" "$R/qwen_rgb_marked_albedo_marked_predictions.jsonl" Qwen3-VL-2B shared_marker_vs_rgb qwen_shared_vs_rgb.json
compare "$RGB/internvl3_5_2b_predictions.jsonl" "$R/internvl_rgb_marked_albedo_marked_predictions.jsonl" InternVL3.5-2B shared_marker_vs_rgb internvl_shared_vs_rgb.json

compare "$ALB/qwen_rgb_albedo_predictions.jsonl" "$R/qwen_rgb_marked_albedo_marked_predictions.jsonl" Qwen3-VL-2B shared_full_vs_crop_pair qwen_shared_vs_crop_pair.json
compare "$ALB/internvl_rgb_albedo_predictions.jsonl" "$R/internvl_rgb_marked_albedo_marked_predictions.jsonl" InternVL3.5-2B shared_full_vs_crop_pair internvl_shared_vs_crop_pair.json

"$PY" - "$R" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
output = {"experiment": "material_constancy_region_alignment_v1", "models": {}}
for model in ["qwen", "internvl"]:
    single_path = root / f"{model}_rgb_marked_albedo_unmarked_predictions.jsonl"
    shared_path = root / f"{model}_rgb_marked_albedo_marked_predictions.jsonl"
    single = {r["sample_id"]: r for r in map(json.loads, single_path.read_text().splitlines())}
    shared = {r["sample_id"]: r for r in map(json.loads, shared_path.read_text().splitlines())}
    ids = sorted(single)
    output["models"][model] = {
        "sample_count": len(ids),
        "answers_changed_by_shared_marker": sum(single[i]["predicted_label"] != shared[i]["predicted_label"] for i in ids),
        "single_marker_label_counts": dict(Counter(single[i]["predicted_label"] or "<invalid>" for i in ids)),
        "shared_marker_label_counts": dict(Counter(shared[i]["predicted_label"] or "<invalid>" for i in ids)),
        "single_summary": json.loads((root / f"{model}_rgb_marked_albedo_unmarked_summary.json").read_text()),
        "shared_summary": json.loads((root / f"{model}_rgb_marked_albedo_marked_summary.json").read_text()),
        "shared_vs_single": json.loads((root / f"{model}_shared_vs_single.json").read_text()),
        "shared_vs_rgb": json.loads((root / f"{model}_shared_vs_rgb.json").read_text()),
        "shared_vs_crop_pair": json.loads((root / f"{model}_shared_vs_crop_pair.json").read_text()),
    }
(root / "region_alignment_overall_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(output, ensure_ascii=False, indent=2))
PY
