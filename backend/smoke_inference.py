"""Run one real local image through all three trained checkpoints."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps
from app.pipeline import FoodPipeline

parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?", help="Path to a meal image")
args = parser.parse_args()

if args.image:
    path = Path(args.image)
else:
    samples = sorted((Path(__file__).resolve().parents[1] / "sample_images").glob("*.jpg"))
    if not samples:
        raise SystemExit("No sample image found. Pass a path explicitly.")
    path = samples[0]

image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
pipeline = FoodPipeline()
print("Model status before load:")
print(json.dumps(pipeline.status(), indent=2, ensure_ascii=False))
print(f"\nAnalyzing: {path}")
result = pipeline.analyze(image)
print(json.dumps(result, indent=2, ensure_ascii=False))
print("\nModel status after load:")
print(json.dumps(pipeline.status(), indent=2, ensure_ascii=False))
