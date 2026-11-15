"""
Day 4: Image classification with a Hugging Face Transformers pipeline, run fully locally.

The model is downloaded from the Hugging Face Hub on the first run and loaded from the
local cache (~/.cache/huggingface) afterwards, so later runs work offline.

Usage (from the project root, with the `local-rag-api` conda env activated):
    python day4_huggingface/classify_image.py data/images/sample.jpg
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline

# Vision Transformer (ViT) pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k (1,000 classes).
MODEL_ID = "google/vit-base-patch16-224"
TOP_K = 5


def pick_device() -> str:
    """Use the Apple Silicon GPU (MPS) when available, otherwise fall back to the CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a local image with a Hugging Face model.")
    parser.add_argument("image", type=Path, help="path to a local JPG or PNG image")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    device = pick_device()
    print(f"Model : {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Image : {args.image}\n")

    # The pipeline bundles preprocessing (resize/normalize), the model and postprocessing
    # (logits -> label names) behind a single call.
    classifier = pipeline("image-classification", model=MODEL_ID, device=device)

    # Convert to RGB so PNGs with an alpha channel or grayscale images also work.
    image = Image.open(args.image).convert("RGB")

    start = time.perf_counter()
    predictions = classifier(image, top_k=TOP_K)
    elapsed = time.perf_counter() - start

    print(f"Top {TOP_K} predictions ({elapsed:.2f} s):")
    for rank, prediction in enumerate(predictions, start=1):
        print(f"{rank}. {prediction['label']:<45} {prediction['score']:.1%}")


if __name__ == "__main__":
    main()