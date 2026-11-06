#!/bin/bash
# ==============================================================================
# Ablation Experiments Runner for MatRoute (Experiment E8)
# Compares:
# 1. Standard Block-Causal (Read-only CORE)
# 2. Soft-gated CORE (--soft_core)
# 3. Stopgrad CORE (--stopgrad_core)
# ==============================================================================
set -e

echo "=== Running MatRoute Ablation Studies (E8) ==="
mkdir -p results/ablation checkpoints/ablation

# 1. Soft-Core Ablation
echo ">>> [Ablation 1] Soft-Core Variant..."
python train_matroute.py \
    --config matroute/configs/full_d128.yaml \
    --epochs 50 \
    --soft_core \
    --checkpoint_dir checkpoints/ablation/soft_core \
    --log_dir results/ablation/soft_core

python test_matroute.py \
    --group seen \
    --model_load checkpoints/ablation/soft_core/checkpoint_latest.pt \
    --output_csv results/ablation/e8_soft_core_seen.csv

# 2. Stopgrad-Core Ablation
echo ">>> [Ablation 2] Stopgrad-Core Variant..."
python train_matroute.py \
    --config matroute/configs/full_d128.yaml \
    --epochs 50 \
    --stopgrad_core \
    --checkpoint_dir checkpoints/ablation/stopgrad_core \
    --log_dir results/ablation/stopgrad_core

python test_matroute.py \
    --group seen \
    --model_load checkpoints/ablation/stopgrad_core/checkpoint_latest.pt \
    --output_csv results/ablation/e8_stopgrad_seen.csv

echo "=== Ablation Suite Complete! Inspect results/ablation/ ==="
