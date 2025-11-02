from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gesture67 import build_mobilenet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileNetV3 checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx_path", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_mobilenet()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size, dtype=torch.float32)
    args.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy, # type: ignore
        args.onnx_path,
        opset_version=args.opset,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )
    if args.verify:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        active = [p for p in providers if p in available]
        if not active:
            raise RuntimeError("no onnxruntime providers are available")
        session = ort.InferenceSession(str(args.onnx_path), providers=active)
        input_name = session.get_inputs()[0].name
        dummy_np = dummy.numpy()
        outputs = session.run(None, {input_name: dummy_np})
        if not outputs or outputs[0].shape[0] != args.batch_size:
            raise RuntimeError("onnx runtime verification failed")


if __name__ == "__main__":
    main()
