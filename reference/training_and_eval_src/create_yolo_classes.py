# 
from pathlib import Path
from ultralytics import YOLO
import json


ROOT = Path(__file__).resolve().parent.parent


MODEL_PATH = (
    ROOT
    / "checkpoints"
    / "merged_yolo26n_seg"
    / "weights"
    / "best.pt"
)


OUTPUT = (
    ROOT
    / "configs"
    / "yolo_classes.json"
)


model = YOLO(
    str(MODEL_PATH)
)


classes = model.names


OUTPUT.parent.mkdir(
    exist_ok=True
)


with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        classes,
        f,
        indent=4,
        ensure_ascii=False
    )


print("Saved:")
print(OUTPUT)


print("\nClasses:")

for k,v in classes.items():
    print(k,":",v)