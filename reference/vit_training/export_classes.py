import os
import json

TRAIN_DIR = "datasets/final_dataset/train"

classes = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])

print("Total Classes:", len(classes))

with open("classes.json", "w", encoding="utf-8") as f:
    json.dump(classes, f, indent=4, ensure_ascii=False)

print("Saved: classes.json")
