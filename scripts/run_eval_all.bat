@echo off
REM ==============================================================================
REM Comprehensive Evaluation Script - Windows Batch
REM ==============================================================================

set MODEL_CKPT=%1
if "%MODEL_CKPT%"=="" set MODEL_CKPT=checkpoints\full_d128\checkpoint_best.pt

echo === Running Full MatRoute Evaluation using: %MODEL_CKPT% ===
if not exist results mkdir results

REM Experiment E1: 11 Seen Problems
echo >>> [E1] Evaluating 11 Seen Problems...
python test_matroute.py ^
    --group seen ^
    --model_load %MODEL_CKPT% ^
    --output_csv results\e1_seen.csv

REM Experiment E2a: 44 Unseen Symmetric Problems (Table 11)
echo >>> [E2a] Evaluating 44 Zero-Shot Symmetric Problems (Table 11)...
python test_matroute.py ^
    --group unseen44 ^
    --model_load %MODEL_CKPT% ^
    --output_csv results\e2_unseen44.csv

REM Experiment E2b: 55 Unseen Asymmetric & PDCVRP Problems (Table 12)
echo >>> [E2b] Evaluating 55 Zero-Shot Asymmetric/PDCVRP Problems (Table 12)...
python test_matroute.py ^
    --group unseen55 ^
    --model_load %MODEL_CKPT% ^
    --output_csv results\e2_unseen55.csv

REM Experiment E6: Funnel Decoder
echo >>> [E6] Running Funnel Decoder Sweep...
python test_matroute.py ^
    --problems cvrp,cvrptw ^
    --level FUNNEL ^
    --model_load %MODEL_CKPT% ^
    --output_csv results\e6_funnel.csv

echo === Evaluation Complete! ===
pause
