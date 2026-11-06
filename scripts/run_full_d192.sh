#!/bin/bash
# ==============================================================================
# MatRoute Full Training (d=192, Primary Main Model)
# 500 epochs x 2000 batches x 128 batch_size
# Estimated time: ~140 hours on RTX 4090 / A100
# Resumable: automatically recovers if interrupted
# ==============================================================================
set -e

echo "=== Starting MatRoute Primary Model Training (d=192) ==="
mkdir -p results/full_d192 checkpoints/full_d192

RESUME_FLAG=""
if [ -f "checkpoints/full_d192/checkpoint_latest.pt" ]; then
    echo "Existing checkpoint detected. Resuming training..."
    RESUME_FLAG="--resume checkpoints/full_d192/checkpoint_latest.pt"
fi

python train_matroute.py \
    --config matroute/configs/full_d192.yaml \
    --checkpoint_dir checkpoints/full_d192 \
    --log_dir results/full_d192 \
    $RESUME_FLAG

echo "=== Training Complete. Running Gate G2 Evaluation on Seen variants ==="
python test_matroute.py \
    --group seen \
    --model_load checkpoints/full_d192/checkpoint_best.pt \
    --output_csv results/full_d192/seen_eval.csv

echo "=== All Done for d=192 ==="
