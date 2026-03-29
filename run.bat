@echo off
REM Model-Nested-Spider Training Runner for Windows

REM Get the script directory (repo root)
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Load environment variables from .env
if exist .env (
    echo Loading environment variables from .env...
    for /f "usebackq delims=" %%a in (.env) do (
        if not "%%a"=="" (
            if not "%%a:~0,1%%" == "#" (
                set "%%a"
            )
        )
    )
    echo Environment variables loaded!
) else (
    echo Warning: .env file not found. Using defaults.
)

REM Default arguments (customize these as needed)
if not defined GPU set GPU=0
if not defined SEED set SEED=1
if not defined TRAIN_DATASET set TRAIN_DATASET=CIFAR10
if not defined TEST_DATASET set TEST_DATASET=CIFAR10
if not defined BATCH_SIZE set BATCH_SIZE=128
if not defined MAX_EPOCH set MAX_EPOCH=50
if not defined LR set LR=0.01
if not defined OPTIMIZER set OPTIMIZER=Adam
if not defined LR_SCHEDULER set LR_SCHEDULER=cosine
if not defined DATA_SUB_URL set DATA_SUB_URL=swin_base_7_checkpoint

REM Print configuration
echo.
echo ========================================
echo Model-Nested-Spider Configuration
echo ========================================
echo GPU: %GPU%
echo Seed: %SEED%
echo Train Dataset: %TRAIN_DATASET%
echo Test Dataset: %TEST_DATASET%
echo Batch Size: %BATCH_SIZE%
echo Max Epoch: %MAX_EPOCH%
echo Learning Rate: %LR%
echo Optimizer: %OPTIMIZER%
echo LR Scheduler: %LR_SCHEDULER%
echo Data SubURL: %DATA_SUB_URL%
if defined PRETRAINED_URL (
    echo Pretrained URL: %PRETRAINED_URL%
)
echo ========================================
echo.

REM Build and run command
set "CMD=python trainer.py ^
    --gpu %GPU% ^
    --seed %SEED% ^
    --train_dataset %TRAIN_DATASET% ^
    --test_dataset %TEST_DATASET% ^
    --batch_size %BATCH_SIZE% ^
    --max_epoch %MAX_EPOCH% ^
    --lr %LR% ^
    --optimizer %OPTIMIZER% ^
    --lr_scheduler %LR_SCHEDULER% ^
    --data_sub_url %DATA_SUB_URL%"

if defined PRETRAINED_URL (
    set "CMD=!CMD! --pretrained_url %PRETRAINED_URL%"
)

REM Add any additional arguments
if not "%~1"=="" (
    set "CMD=!CMD! %*"
)

echo Starting training...
echo Command: %CMD%
echo.
%CMD%

endlocal
