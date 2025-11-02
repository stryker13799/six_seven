# 67 Gesture Detector

Captures training data, trains a MobileNetV3 detector, exports to ONNX, and runs a live webcam demo with state machine detection and meme effects when the "67" hand pose is detected.

## quick start

1. Install dependencies
   ```
   pip install -r requirements.txt
   ```
2. Prepare folders:
   ```
   data\images\positive
   data\images\negative
   ```
3. Record samples:
   ```
   python scripts\capture_dataset.py --output-root data --camera-index 0
   ```

## training

### Automated Training Pipeline (Recommended)

Use the automated scripts to train and export the model in one command:

**Windows (PowerShell/CMD):**
```powershell
.\train_and_export.bat
```

**Linux/WSL/Git Bash:**
```bash
chmod +x train_and_export.sh
./train_and_export.sh
```

These scripts will:
- Train the MobileNetV3 model with default parameters
- Save the checkpoint to `artifacts/mobilenet_v3.pth`
- Export the trained model to ONNX format
- Verify the ONNX export
- Display progress and summary

To customize training parameters, edit the configuration section at the top of the script.

### Manual Training

MobileNetV3:
```powershell
python scripts\train_mobilenet.py --image-root data\images --epochs 20 --batch-size 32
python scripts\export_onnx.py --checkpoint artifacts\mobilenet_v3.pth --onnx_path artifacts\mobilenet_v3.onnx --verify
```

## live inference

```powershell
python scripts\live_demo.py --mirror
```

The demo uses a state machine that requires 8 consecutive frames of 67 detection before triggering. When 67 is detected, it overlays `resource\67.jpg` and loops `resource\67.mp3` with deepfried visual effects.

## data layout

```
project/
├─ data/
│  └─ images/
│     ├─ positive/
│     └─ negative/
├─ artifacts/
├─ resource/
│  ├─ 67.jpg
│  └─ 67.mp3
└─ scripts/
```

## notes

* Dataset split: 70/15/15 train/val/test with fixed seed
* State machine prevents false positives from single frame glitches
* ONNX Runtime prefers CUDA when available
