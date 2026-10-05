import os
import shutil
from tqdm import tqdm

OVERHEAD_DIR = "datasets/nutrition_dataset/imagery/realsense_overhead"
SIDE_DIR = "datasets/nutrition_dataset/imagery/side_angles"

OUTPUT_DIR = "datasets/sam2_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

dishes = sorted(os.listdir(OVERHEAD_DIR))

print("Total dishes:", len(dishes))

SIDE_LIMIT = 25  # change here easily

for dish_id in tqdm(dishes, desc="Building Dataset"):

    src_overhead = os.path.join(OVERHEAD_DIR, dish_id)
    src_side = os.path.join(SIDE_DIR, dish_id)

    if not os.path.exists(src_overhead) or not os.path.exists(src_side):
        continue

    dst_dish = os.path.join(OUTPUT_DIR, dish_id)
    os.makedirs(dst_dish, exist_ok=True)

    # -----------------
    # RGB image
    # -----------------
    rgb_path = os.path.join(src_overhead, "rgb.png")

    if os.path.exists(rgb_path):
        shutil.copy2(rgb_path, os.path.join(dst_dish, "rgb.png"))

    # -----------------
    # Side views (25 frames)
    # -----------------
    side_frames = sorted([
        f for f in os.listdir(src_side)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:SIDE_LIMIT]

    for frame in side_frames:

        src = os.path.join(src_side, frame)
        dst = os.path.join(dst_dish, frame)

        if os.path.exists(src):
            shutil.copy2(src, dst)

print("\nDONE ✅")
print("Dataset saved to:", OUTPUT_DIR)