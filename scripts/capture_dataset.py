from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam capture for 67 gesture dataset")
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--auto-interval", type=float, default=0.01)
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    (root / "images" / "positive").mkdir(parents=True, exist_ok=True)
    (root / "images" / "negative").mkdir(parents=True, exist_ok=True)


def save_capture(frame: np.ndarray, label: str, root: Path) -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_id = f"{timestamp}_{uuid4().hex[:8]}"
    image_path = root / "images" / label / f"{file_id}.jpg"
    cv2.imwrite(str(image_path), frame)


def main() -> None:
    args = parse_args()
    root = args.output_root
    ensure_dirs(root)
    
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("unable to open camera")
    
    auto_mode = False
    last_label = "positive"
    last_auto = 0.0
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue
            
            if args.mirror:
                frame = cv2.flip(frame, 1)
            
            display = frame.copy()
            status = f"mode: {'auto' if auto_mode else 'manual'} | label: {last_label}"
            cv2.putText(display, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "p=positive n=negative space=toggle auto q=quit", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("67 capture", display)
            key = cv2.waitKey(1) & 0xFF
            
            label_to_save = None
            if key == ord("p"):
                last_label = "positive"
                label_to_save = last_label
            elif key == ord("n"):
                last_label = "negative"
                label_to_save = last_label
            elif key == ord(" "):
                auto_mode = not auto_mode
            elif key == ord("q"):
                break
            
            if label_to_save:
                save_capture(frame, label_to_save, root)
                last_auto = time.time()
            
            if auto_mode and time.time() - last_auto >= args.auto_interval:
                save_capture(frame, last_label, root)
                last_auto = time.time()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
