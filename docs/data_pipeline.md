# data and training pipeline

## capture

`scripts/capture_dataset.py` opens the webcam, runs MediaPipe Hands, and waits for key presses:

* `p`: save as positive ("67" gesture)
* `n`: save as negative (anything else)
* `space`: toggle auto-save (grabs frames every second with the current label)
* `q`: quit

Each saved frame lands in `data/images/<label>/timestamp_id.jpg`. MediaPipe supplies up to two hands per frame; the script stores separate left and right confidences plus 63 xyz values per hand (126 floats total). Missing hands are zero-filled so negatives with a single hand remain usable. Auto-save uses the most recent manual label so you can stream multiple frames per gesture without spamming the keyboard.

## augmentation

Image training uses random resized crops, horizontal flips, color jitter, and light gaussian noise through Albumentations. Validation and test flows run a deterministic resize+center crop. Landmarks get augmented by random jitter, dropout on a few joints, and global rotation noise inside the landmark training script.

## splits

Both training scripts use the same 70/15/15 split determined by a reproducible seed. There is no persistent split file; rerunning the training scripts re-splits the dataset using the same seed so long as the dataset contents remain stable.

## training

* MobileNetV3: fine-tuned end-to-end with AdamW, cosine learning rate schedule, label smoothing cross-entropy, and mixed precision if CUDA is available.
* Landmark MLP: two-layer fully connected network trained on the 126-float vectors (left+right) with Adam, dropout, and early stopping on the validation set.

## export and inference

The MobileNet weights export to ONNX (`artifacts/mobilenet_v3.onnx`). The exporter validates correctness by running an ONNX Runtime inference pass with providers ordered `CUDAExecutionProvider`, then `CPUExecutionProvider`. The live demo loads the ONNX model. When CUDA is absent, everything transparently drops to CPU.
