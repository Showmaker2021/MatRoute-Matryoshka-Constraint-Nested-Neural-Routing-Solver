@echo off
REM ==============================================================================
REM MatRoute Pilot Training (50 epochs) - Windows Batch
REM ==============================================================================

echo === Starting MatRoute Pilot Training (50 epochs) ===
if not exist results\pilot mkdir results\pilot
if not exist checkpoints\pilot mkdir checkpoints\pilot

python train_matroute.py ^
    --config matroute/configs/full_d128.yaml ^
    --epochs 50 ^
    --checkpoint_dir checkpoints/pilot ^
    --log_dir results/pilot

echo === Running Gate G1 Pilot Evaluation ===
python test_matroute.py ^
    --group seen ^
    --model_load checkpoints/pilot/checkpoint_latest.pt ^
    --max_instances 100 ^
    --output_csv results/pilot_eval.csv

echo === Pilot Finished! Inspect results/pilot_eval.csv against docs/GATES.md ===
pause
