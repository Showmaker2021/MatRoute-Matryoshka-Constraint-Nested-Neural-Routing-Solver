#!/bin/bash
# ==============================================================================
# MatRoute Pilot Training (50 epochs)
# Estimated time: ~10 hours on RTX 4090 / A100
# ==============================================================================
set -e

echo "=== Starting MatRoute Pilot Training (50 epochs) ==="
mkdir -p results/pilot checkpoints/pilot

python train_matroute.py \
    --config matroute/configs/full_d128.yaml \
    --epochs 50 \
    --checkpoint_dir checkpoints/pilot \
    --log_dir results/pilot

echo "=== Running Gate G1 Pilot Evaluation ==="
python test_matroute.py \
    --group seen \
    --model_load checkpoints/pilot/checkpoint_latest.pt \
    --max_instances 100 \
    --output_csv results/pilot_eval.csv

echo "=== Pilot Finished! Inspect results/pilot_eval.csv against docs/GATES.md ==="
