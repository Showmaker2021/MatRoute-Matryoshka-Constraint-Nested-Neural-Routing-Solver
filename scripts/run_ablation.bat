@echo off
REM ==============================================================================
REM Ablation Experiments Runner - Windows Batch
REM ==============================================================================

echo === Running MatRoute Ablation Studies (E8) ===
if not exist results\ablation mkdir results\ablation
if not exist checkpoints\ablation mkdir checkpoints\ablation

echo >>> [Ablation 1] Soft-Core Variant...
python train_matroute.py ^
    --config matroute/configs/full_d128.yaml ^
    --epochs 50 ^
    --soft_core ^
    --checkpoint_dir checkpoints\ablation\soft_core ^
    --log_dir results\ablation\soft_core

python test_matroute.py ^
    --group seen ^
    --model_load checkpoints\ablation\soft_core\checkpoint_latest.pt ^
    --output_csv results\ablation\e8_soft_core_seen.csv

echo >>> [Ablation 2] Stopgrad-Core Variant...
python train_matroute.py ^
    --config matroute/configs/full_d128.yaml ^
    --epochs 50 ^
    --stopgrad_core ^
    --checkpoint_dir checkpoints\ablation\stopgrad_core ^
    --log_dir results\ablation\stopgrad_core

python test_matroute.py ^
    --group seen ^
    --model_load checkpoints\ablation\stopgrad_core\checkpoint_latest.pt ^
    --output_csv results\ablation\e8_stopgrad_seen.csv

echo === Ablation Done! ===
pause
