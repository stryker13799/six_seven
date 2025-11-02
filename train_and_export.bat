@echo off
REM Automated Training and ONNX Export Script for Windows
REM This script trains the MobileNetV3 model and exports it to ONNX format

setlocal enabledelayedexpansion

echo ==========================================
echo Starting Automated Training Pipeline
echo ==========================================
echo.

REM Configuration
set CHECKPOINT_PATH=artifacts/mobilenet_v3.pth
set ONNX_PATH=artifacts/mobilenet_v3.onnx
set METRICS_PATH=artifacts/mobilenet_v3_metrics.json

REM Training parameters
set EPOCHS=10
set BATCH_SIZE=64
set LEARNING_RATE=3e-4
set SEED=42

echo Step 1: Training MobileNetV3 model...
echo --------------------------------------
echo Epochs: %EPOCHS%
echo Batch Size: %BATCH_SIZE%
echo Learning Rate: %LEARNING_RATE%
echo.

python scripts\train_mobilenet.py --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --seed %SEED% --checkpoint %CHECKPOINT_PATH% --metrics-path %METRICS_PATH%

if %errorlevel% neq 0 (
    echo Error: Training failed!
    exit /b 1
)

echo.
echo Training completed successfully!
echo Checkpoint saved to: %CHECKPOINT_PATH%
echo Metrics saved to: %METRICS_PATH%
echo.

echo Step 2: Exporting model to ONNX format...
echo ------------------------------------------

python scripts\export_onnx.py --checkpoint %CHECKPOINT_PATH% --onnx_path %ONNX_PATH% --verify

if %errorlevel% neq 0 (
    echo Error: ONNX export failed!
    exit /b 1
)

echo.
echo ONNX export completed successfully!
echo ONNX model saved to: %ONNX_PATH%
echo.

echo ==========================================
echo Pipeline Completed Successfully!
echo ==========================================
echo.
echo Summary:
echo   - Model checkpoint: %CHECKPOINT_PATH%
echo   - ONNX model: %ONNX_PATH%
echo   - Training metrics: %METRICS_PATH%
echo.

endlocal
