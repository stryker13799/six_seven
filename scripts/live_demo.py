from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pygame
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ONNX_PATH = r"C:\Users\Ali\Desktop\six_seven\artifacts\mobilenet_v3.onnx"
MEME_IMAGE = r"C:\Users\Ali\Desktop\six_seven\resource\67.jpg"
MEME_AUDIO = r"C:\Users\Ali\Desktop\six_seven\resource\67.mp3"
MIN_CONFIDENCE = 0.6
CONSECUTIVE_FRAMES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live 67 detector demo")
    parser.add_argument("--onnx", type=Path, default=Path(ONNX_PATH))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--consecutive", type=int, default=CONSECUTIVE_FRAMES)
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()


def load_onnx_session(path: Path) -> tuple[ort.InferenceSession, str]:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    active = [p for p in providers if p in available]
    if not active:
        raise RuntimeError("no onnxruntime providers available")
    session = ort.InferenceSession(str(path), providers=active)
    input_name = session.get_inputs()[0].name
    return session, input_name


def preprocess_frame(frame, transforms) -> np.ndarray:
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    tensor = transforms(pil_image)
    return tensor.unsqueeze(0).numpy()


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        raise FileNotFoundError(f"missing onnx model: {args.onnx}")
    
    session, input_name = load_onnx_session(args.onnx)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    eval_tf = weights.transforms()
    
    pygame.mixer.init()
    meme_img = None
    if Path(MEME_IMAGE).exists():
        meme_img = cv2.imread(MEME_IMAGE)
    
    detection_buffer = deque(maxlen=args.consecutive)
    is_67_active = False
    
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("unable to open camera")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            
            if args.mirror:
                frame = cv2.flip(frame, 1)
            
            ort_input = preprocess_frame(frame, eval_tf)
            logits = session.run(None, {input_name: ort_input})[0]
            probs = softmax_np(logits)
            positive_prob = float(probs[0, 1])
            
            is_67_frame = positive_prob >= args.threshold
            detection_buffer.append(is_67_frame)
            
            sequence_detected = len(detection_buffer) == args.consecutive and all(detection_buffer)
            
            if sequence_detected and not is_67_active:
                is_67_active = True
                if Path(MEME_AUDIO).exists():
                    try:
                        pygame.mixer.music.load(MEME_AUDIO)
                        pygame.mixer.music.play(-1)
                    except:
                        pass
            elif not sequence_detected and is_67_active:
                is_67_active = False
                pygame.mixer.music.stop()
            
            if is_67_active and meme_img is not None:
                h, w = frame.shape[:2]
                meme_resized = cv2.resize(meme_img, (w, h))
                alpha = 0.7
                frame = cv2.addWeighted(frame, 1 - alpha, meme_resized, alpha, 0)
                
                frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
            
            color = (0, 255, 0) if is_67_active else ((255, 165, 0) if is_67_frame else (0, 0, 255))
            status = "67 DETECTED" if is_67_active else ("detecting..." if is_67_frame else "waiting")
            
            cv2.putText(frame, status, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"conf: {positive_prob:.2f}", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"buffer: {sum(detection_buffer)}/{len(detection_buffer)}", (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("67 live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
