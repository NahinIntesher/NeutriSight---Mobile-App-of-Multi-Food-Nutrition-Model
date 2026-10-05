import os
import shutil
import random
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================

source_dataset = r"datasets/combined_dataset"

output_dataset = r"datasets/final_dataset"

# =========================
# SPLIT RATIOS
# =========================

train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

# =========================
# CREATE OUTPUT FOLDERS
# =========================

for split in ["train", "val", "test"]:

    os.makedirs(os.path.join(output_dataset, split), exist_ok=True)

# =========================
# PROCESS EACH CLASS
# =========================

for class_name in os.listdir(source_dataset):

    class_path = os.path.join(source_dataset, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    # Shuffle images
    random.shuffle(images)

    # First split train vs temp
    train_images, temp_images = train_test_split(
        images,
        test_size=(1 - train_ratio),
        random_state=42
    )

    # Split temp into val and test
    val_images, test_images = train_test_split(
        temp_images,
        test_size=0.50,
        random_state=42
    )

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    # =========================
    # COPY IMAGES
    # =========================

    for split_name, split_images in splits.items():

        split_class_dir = os.path.join(
            output_dataset,
            split_name,
            class_name
        )

        os.makedirs(split_class_dir, exist_ok=True)

        for image_name in split_images:

            src = os.path.join(class_path, image_name)

            dst = os.path.join(split_class_dir, image_name)

            shutil.copy2(src, dst)

    print(f"Done: {class_name}")

print("\nDataset splitting completed.")