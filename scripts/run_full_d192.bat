@echo off
REM ==============================================================================
REM MatRoute Primary Model Training (d=192) - Windows Batch
REM ==============================================================================

echo === Starting MatRoute Primary Model Training (d=192) ===
if not exist results\full_d192 mkdir results\full_d192
if not exist checkpoints\full_d192 mkdir checkpoints\full_d192

set RESUME_ARG=
if exist checkpoints\full_d192\checkpoint_latest.pt (
    echo Existing checkpoint found. Resuming...
    set RESUME_ARG=--resume checkpoints\full_d192\checkpoint_latest.pt
)

python train_matroute.py ^
    --config matroute/configs/full_d192.yaml ^
    --checkpoint_dir checkpoints/full_d192 ^
    --log_dir results/full_d192 ^
    %RESUME_ARG%

echo === Training Complete. Evaluating Seen variants ===
python test_matroute.py ^
    --group seen ^
    --model_load checkpoints\full_d192\checkpoint_best.pt ^
    --output_csv results\full_d192\seen_eval.csv

echo === Done! ===
pause
