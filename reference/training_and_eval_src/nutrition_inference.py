# src/nutrition_inference.py

from pathlib import Path
import json
import cv2
import torch
import timm
from PIL import Image
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "checkpoints"
    / "nutrition5k_vit"
    / "best_model.pth"
)

IMAGE_PATH = ROOT / "src" / "yolo_vit_output.jpg"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

TARGETS = [
    "calories",
    "mass",
    "fat",
    "carb",
    "protein"
]


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("=" * 80)
print("NUTRITION5K ViT INFERENCE")
print("=" * 80)

print(f"\nDevice: {DEVICE}")
print("Loading Nutrition model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

mean = torch.tensor(
    checkpoint["mean"],
    dtype=torch.float32
)

std = torch.tensor(
    checkpoint["std"],
    dtype=torch.float32
)

targets = checkpoint.get(
    "targets",
    TARGETS
)


model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=len(targets)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Nutrition model loaded.")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


x = transform(
    image
).unsqueeze(0).to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning nutrition prediction...")

with torch.no_grad():

    with torch.amp.autocast(
        "cuda",
        enabled=(
            DEVICE == "cuda"
        )
    ):

        output = model(x)


prediction = (
    output[0].cpu()
    * std
    + mean
)

prediction = prediction.numpy()


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("NUTRITION PREDICTION")
print("=" * 80)

print(
    f"Calories : {prediction[0]:.2f} kcal"
)

print(
    f"Mass     : {prediction[1]:.2f} g"
)

print(
    f"Fat      : {prediction[2]:.2f} g"
)

print(
    f"Carbs    : {prediction[3]:.2f} g"
)

print(
    f"Protein  : {prediction[4]:.2f} g"
)

print("=" * 80)
print("COMPLETED")
print("=" * 80)