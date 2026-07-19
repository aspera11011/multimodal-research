# 共同区域标记实验结果

## 研究问题

此前直接输入 RGB crop 与 albedo crop 没有稳定改善材质判断。本实验只检查一个可能原因：冻结 VLM 是否因为不能明确对应 RGB 与 albedo 中的同一区域，而没有利用 albedo。

## 实验设计

- 固定原有 66 个区域、330 个跨光照样本和 11 个材质类别。
- 使用 Qwen3-VL-2B-Instruct 与 InternVL3.5-2B-HF，均不训练。
- 单框条件：完整 RGB 中用红框标记目标，配准 albedo 不画框。
- 共同框条件：完整 RGB 与配准 albedo 都在相同坐标画同样红框。
- 两组使用相同图像、问题、候选标签和解码设置，唯一实验差异是 albedo 中是否有对应红框。
- 指标为样本准确率、66 个区域中发生跨光照答案翻转的比例，以及区域级配对 bootstrap 95% CI。

## 主要结果

| 模型 | 单框准确率 | 共同框准确率 | 准确率差值 | 单框翻转率 | 共同框翻转率 | 翻转率差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | 45.76% | 47.58% | +1.82 pp，CI [-1.82, 5.76] | 54.55% | 51.52% | -3.03 pp，CI [-10.61, 4.55] |
| InternVL | 48.79% | 51.21% | +2.42 pp，CI [-2.42, 6.67] | 51.52% | 45.45% | -6.06 pp，CI [-15.15, 3.03] |

共同框改变了 Qwen 的 61/330 个答案和 InternVL 的 60/330 个答案，说明模型看到了标记差异；两个模型的准确率和稳定性都朝有利方向变化，但所有“共同框相对单框”的 95% CI 都跨 0，尚不能认定改善稳定存在。

## 与既有基线的关系

- Qwen 共同框准确率 47.58%，低于原始 RGB 的 60.30%，差值 -12.73 pp，95% CI [-20.30, -5.75]；因此共同框不能恢复 Qwen 的基线能力。
- InternVL 共同框准确率 51.21%，与原始 RGB 的 53.03% 无显著差异；翻转率从 66.67% 降至 45.45%，但这不是共同框的纯效果，因为完整场景、RGB 框和 albedo 接口都同时变化。
- 相对原有裁剪 RGB+albedo，Qwen 共同框准确率仍显著更差；InternVL 准确率相同。故不能把完整场景共同框视作通用改进。

## 当前判断

共同区域标记呈现小幅、跨模型同方向趋势，但没有通过预设的“两个模型均稳定改善”门槛。它最多说明区域对应可能是接口问题的一小部分，不能说明冻结 VLM 已学会利用 albedo。零训练视觉标记路线继续扩大实验的价值较低；若继续该问题，应转向训练时的一致性约束或轻量对齐模块，并将 albedo 作为辅助监督，而不是继续增加提示元素。

## 可复现材料

- 配置：`configs/material_constancy_region_alignment_v1.json`
- 构建脚本：`scripts/build_material_region_alignment.py`
- 推理入口：`scripts/run_material_region_alignment.sh`
- 统计入口：`scripts/analyze_material_region_alignment.sh`
- 固定清单：`experiments/manifests/material_constancy_region_alignment_v1/`
- 逐样本预测与汇总：`results/quantitative/material_constancy_region_alignment_v1/`
- 构建、推理和分析日志：`experiments/logs/material_constancy/`

本地与远程总汇总 `region_alignment_overall_summary.json` 的 SHA-256 均为 `38b6c594e1f3bfe46cb41c48504d4649eece30cad3ff24d6aae29d0f9dd0d132`。
