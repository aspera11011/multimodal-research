# SGNet RGB-D-D 16× baseline and shift gate

## Purpose

Recover a traceable SGNet baseline on the existing 405-pair RGB-D-D `test2` set, then measure whether small RGB-only shifts create reproducible depth-boundary errors.

## Boundary

- This is a cross-dataset synthetic 16× pilot using the NYU-trained SGNet checkpoint.
- RMSE/MAE are reported in 8-bit depth levels after the historical 6-pixel crop.
- It is not the official RGB-D-D real-sensor 4× protocol.
- No C2PD code or training is started until the shift finding passes the gate.

## Required records

- experiment ID and Git commit;
- checkpoint SHA-256;
- 405 RGB/depth filename pairs;
- Python/PyTorch/CUDA and GPU;
- command, stdout, per-sample JSONL and summary JSON;
- clean and 1/2/4-pixel horizontal/vertical shifts.

## Execution

Use environment variables for server-local paths:

```bash
python scripts/evaluate_sgnet_rgbdd_x16.py \
  --sgnet-dir "$SGNET_DIR" \
  --data-root "$RGBDD_TEST2" \
  --checkpoint "$SGNET_X16_CHECKPOINT" \
  --output-dir "$EXPERIMENT_DIR/clean" \
  --device cuda:0
```

Run `--max-samples 1` first. Expand to 405 only after the sample name, tensor shapes and clean metrics pass inspection.

## Gate

Proceed to a C2PD deformation/continuity audit only when at least two small-shift levels consistently worsen RMSE, boundary RMSE or false-edge rate, with a paired-bootstrap 95% confidence interval excluding zero. Otherwise stop the misalignment route.
