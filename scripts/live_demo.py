from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
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
MIN_CONFIDENCE = 0.8
DEFAULT_MIN_ACTIVE_DURATION = 0.5
DEFAULT_TOLERANCE_SECONDS = 1.0
DEFAULT_COOLDOWN_SECONDS = 1.0


class DetectionState(Enum):
    IDLE = auto()
    DETECTING = auto()
    ACTIVE = auto()
    COOLDOWN = auto()


@dataclass
class DetectionStateMachine:
    min_active_duration: float
    tolerance_seconds: float
    cooldown_seconds: float
    state: DetectionState = DetectionState.IDLE
    detect_start: float | None = None
    last_positive: float | None = None
    active_since: float | None = None
    cooldown_until: float | None = None

    def reset_detection_window(self) -> None:
        self.detect_start = None
        self.last_positive = None

    def update(self, is_positive: bool, now: float | None = None) -> DetectionState:
        if now is None:
            now = time.monotonic()

        if self.state is DetectionState.COOLDOWN:
            if self.cooldown_until is not None and now >= self.cooldown_until:
                self.state = DetectionState.IDLE
                self.cooldown_until = None
                self.active_since = None
                self.reset_detection_window()
            else:
                return self.state

        if self.state is DetectionState.IDLE:
            if is_positive:
                self.state = DetectionState.DETECTING
                self.detect_start = now
                self.last_positive = now
            return self.state

        if self.state is DetectionState.DETECTING:
            if is_positive:
                self.last_positive = now
            elif self.last_positive is not None and now - self.last_positive > self.tolerance_seconds:
                self.state = DetectionState.IDLE
                self.reset_detection_window()
                return self.state

            if (
                self.detect_start is not None
                and now - self.detect_start >= self.min_active_duration
                and self.last_positive is not None
                and now - self.last_positive <= self.tolerance_seconds
            ):
                self.state = DetectionState.ACTIVE
                self.active_since = now
                return self.state

            return self.state

        if self.state is DetectionState.ACTIVE:
            if is_positive:
                self.last_positive = now
                return self.state

            if self.last_positive is None or now - self.last_positive > self.tolerance_seconds:
                self.state = DetectionState.COOLDOWN
                self.cooldown_until = now + self.cooldown_seconds
                self.reset_detection_window()
            return self.state

        return self.state

    def detection_elapsed(self, now: float) -> float | None:
        if self.detect_start is None:
            return None
        return now - self.detect_start

    def active_elapsed(self, now: float) -> float | None:
        if self.active_since is None:
            return None
        return now - self.active_since

    def time_since_last_positive(self, now: float) -> float | None:
        if self.last_positive is None:
            return None
        return now - self.last_positive

    def cooldown_remaining(self, now: float) -> float | None:
        if self.state is not DetectionState.COOLDOWN or self.cooldown_until is None:
            return None
        return max(0.0, self.cooldown_until - now)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live 67 detector demo")
    parser.add_argument("--onnx", type=Path, default=Path(ONNX_PATH))
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--min_active_duration", type=float, default=DEFAULT_MIN_ACTIVE_DURATION)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_SECONDS)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_SECONDS)
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
    meme_img = cv2.imread(MEME_IMAGE) if Path(MEME_IMAGE).exists() else None
    meme_audio_path = Path(MEME_AUDIO)
    audio_available = meme_audio_path.exists()

    state_machine = DetectionStateMachine(
        min_active_duration=max(0.0, args.min_active_duration),
        tolerance_seconds=max(0.0, args.tolerance),
        cooldown_seconds=max(0.0, args.cooldown),
    )
    prev_state = state_machine.state
    
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
            now = time.monotonic()
            state = state_machine.update(is_67_frame, now)

            if prev_state != state:
                if state is DetectionState.ACTIVE:
                    if audio_available:
                        try:
                            pygame.mixer.music.load(str(meme_audio_path))
                            pygame.mixer.music.play(-1)
                        except pygame.error:
                            pass
                elif prev_state is DetectionState.ACTIVE:
                    pygame.mixer.music.stop()
                prev_state = state

            if state is DetectionState.ACTIVE and meme_img is not None:
                h, w = frame.shape[:2]
                meme_resized = cv2.resize(meme_img, (w, h))
                alpha = 0.7
                frame = cv2.addWeighted(frame, 1 - alpha, meme_resized, alpha, 0)
                
                frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
            
            status_colors = {
                DetectionState.IDLE: (0, 0, 255),
                DetectionState.DETECTING: (255, 165, 0),
                DetectionState.ACTIVE: (0, 255, 0),
                DetectionState.COOLDOWN: (128, 0, 128),
            }
            status_labels = {
                DetectionState.IDLE: "waiting",
                DetectionState.DETECTING: "arming",
                DetectionState.ACTIVE: "67 detected",
                DetectionState.COOLDOWN: "cooldown",
            }

            color = status_colors[state]
            cv2.putText(frame, status_labels[state], (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"conf: {positive_prob:.2f}", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            line_y = 110
            if state is DetectionState.DETECTING:
                arming_elapsed = state_machine.detection_elapsed(now) or 0.0
                cv2.putText(
                    frame,
                    f"arming: {arming_elapsed:.2f}/{state_machine.min_active_duration:.2f}s",
                    (12, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                line_y += 30
            elif state is DetectionState.ACTIVE:
                active_elapsed = state_machine.active_elapsed(now) or 0.0
                cv2.putText(
                    frame,
                    f"active: {active_elapsed:.2f}s",
                    (12, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                line_y += 30
            elif state is DetectionState.COOLDOWN:
                cooldown_remaining = state_machine.cooldown_remaining(now) or 0.0
                cv2.putText(
                    frame,
                    f"cooldown: {cooldown_remaining:.2f}s",
                    (12, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                line_y += 30

            if state in (DetectionState.DETECTING, DetectionState.ACTIVE) and state_machine.tolerance_seconds > 0:
                since_positive = state_machine.time_since_last_positive(now)
                if since_positive is not None:
                    tolerance_left = max(0.0, state_machine.tolerance_seconds - since_positive)
                    cv2.putText(
                        frame,
                        f"tolerance left: {tolerance_left:.2f}s",
                        (12, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (160, 160, 160),
                        1,
                    )
            
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
