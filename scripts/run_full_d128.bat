@echo off
REM ==============================================================================
REM MatRoute Full Training (d=128) - Windows Batch
REM ==============================================================================

echo === Starting MatRoute Full Training (d=128) ===
if not exist results\full_d128 mkdir results\full_d128
if not exist checkpoints\full_d128 mkdir checkpoints\full_d128

set RESUME_ARG=
if exist checkpoints\full_d128\checkpoint_latest.pt (
    echo Existing checkpoint found. Resuming...
    set RESUME_ARG=--resume checkpoints\full_d128\checkpoint_latest.pt
)

python train_matroute.py ^
    --config matroute/configs/full_d128.yaml ^
    --checkpoint_dir checkpoints/full_d128 ^
    --log_dir results/full_d128 ^
    %RESUME_ARG% ^
    %*

echo === Training Complete. Evaluating Seen variants ===
python test_matroute.py ^
    --group seen ^
    --model_load checkpoints/full_d128/checkpoint_best.pt ^
    --output_csv results/full_d128/seen_eval.csv

echo === Done! ===
pause
