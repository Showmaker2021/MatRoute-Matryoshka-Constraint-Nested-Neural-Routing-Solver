#!/bin/bash
# ==============================================================================
# MatRoute Full Training (d=128, Parameter-matched baseline)
# 500 epochs x 2000 batches x 128 batch_size
# Estimated time: ~95 hours on RTX 4090 / A100
# Resumable: automatically recovers if interrupted
# ==============================================================================
set -e

echo "=== Starting MatRoute Full Training (d=128) ==="
mkdir -p results/full_d128 checkpoints/full_d128

RESUME_FLAG=""
if [ -f "checkpoints/full_d128/checkpoint_latest.pt" ]; then
    echo "Existing checkpoint detected. Resuming training..."
    RESUME_FLAG="--resume checkpoints/full_d128/checkpoint_latest.pt"
fi

python train_matroute.py \
    --config matroute/configs/full_d128.yaml \
    --checkpoint_dir checkpoints/full_d128 \
    --log_dir results/full_d128 \
    $RESUME_FLAG \
    "$@"

echo "=== Training Complete. Running Gate G2 Evaluation on Seen variants ==="
python test_matroute.py \
    --group seen \
    --model_load checkpoints/full_d128/checkpoint_best.pt \
    --output_csv results/full_d128/seen_eval.csv

echo "=== All Done for d=128 ==="
