#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="/home/xjy/multimodal-research"
PY="/usr/bin/python3"
export PYTHONPATH="$ROOT/models/python_pkgs${PYTHONPATH:+:$PYTHONPATH}"
MANIFEST="$ROOT/experiments/manifests/material_constancy_region_alignment_v1/material_constancy_region_alignment_manifest.jsonl"
RESULTS="$ROOT/results/quantitative/material_constancy_region_alignment_v1"
LOGS="$ROOT/experiments/logs/material_constancy"
QWEN="$ROOT/models/checkpoints/Qwen3-VL-2B-Instruct"
INTERN="$ROOT/models/checkpoints/InternVL3_5-2B-HF"
mkdir -p "$RESULTS" "$LOGS"

if [[ "$MODE" == "smoke" ]]; then
  LIMIT_ARGS=(--limit 10)
  SUFFIX="_smoke"
elif [[ "$MODE" == "full" ]]; then
  LIMIT_ARGS=()
  SUFFIX=""
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

PREFIX_SINGLE='The first image is a full RGB scene with a red rectangle marking the target surface. The second image is the aligned estimated albedo without a rectangle. Use the target at the same spatial location in both images. Classify the target surface material. '
PREFIX_SHARED='The first image is a full RGB scene and the second is its aligned estimated albedo. The same red rectangle marks the target surface in both images. Classify the marked surface material. '

run_one() {
  local runner="$1" model="$2" model_tag="$3" condition="$4" second_key="$5" prefix="$6"
  local output="$RESULTS/${model_tag}_${condition}_predictions${SUFFIX}.jsonl"
  local log="$LOGS/${model_tag}_${condition}${SUFFIX}.log"
  CUDA_VISIBLE_DEVICES=3 "$PY" "$ROOT/scripts/$runner" \
    --manifest "$MANIFEST" \
    --model "$model" \
    --output "$output" \
    --image-key rgb_marked_path \
    --secondary-image-key "$second_key" \
    --condition-name "$condition" \
    --prompt-prefix "$prefix" \
    "${LIMIT_ARGS[@]}" >"$log" 2>&1
  tail -n 2 "$log"
}

nvidia-smi -i 3 --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
run_one run_qwen_material_gate.py "$QWEN" qwen rgb_marked_albedo_unmarked albedo_full_path "$PREFIX_SINGLE"
run_one run_qwen_material_gate.py "$QWEN" qwen rgb_marked_albedo_marked albedo_marked_path "$PREFIX_SHARED"
run_one run_internvl_material_gate.py "$INTERN" internvl rgb_marked_albedo_unmarked albedo_full_path "$PREFIX_SINGLE"
run_one run_internvl_material_gate.py "$INTERN" internvl rgb_marked_albedo_marked albedo_marked_path "$PREFIX_SHARED"
echo "region-alignment $MODE complete"
