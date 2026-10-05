from PIL import Image
import os
from tqdm import tqdm

ROOT = "datasets/yolo_dataset"

removed = 0

for split in ["train", "val"]:

    img_dir = os.path.join(ROOT, "images", split)
    lbl_dir = os.path.join(ROOT, "labels", split)

    images = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    for file in tqdm(images, desc=f"Checking {split}"):

        img_path = os.path.join(img_dir, file)

        try:
            with Image.open(img_path) as img:
                img.verify()

            with Image.open(img_path) as img:
                img.load()

        except Exception:

            label_path = os.path.join(
                lbl_dir,
                os.path.splitext(file)[0] + ".txt"
            )

            print(f"\nREMOVED: {img_path}")

            if os.path.exists(img_path):
                os.remove(img_path)

            if os.path.exists(label_path):
                os.remove(label_path)

            removed += 1

print("\n====================")
print(f"Removed {removed} corrupt images")
print("====================")