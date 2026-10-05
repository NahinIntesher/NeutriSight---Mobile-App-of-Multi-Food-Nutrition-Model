# src/yolo_vit_eval.py

from pathlib import Path
import json
import cv2
import torch
import timm
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent


YOLO_MODEL = (
    ROOT
    / "checkpoints"
    / "merged_yolo26n_seg"
    / "weights"
    / "best.pt"
)


VIT_MODEL = (
    ROOT
    / "checkpoints"
    / "merged_yolo26n_seg"
    / "weights"
    / "best_model.pth"
)


IMAGE_DIR = (
    ROOT
    / "datasets"
    / "merged_dataset"
    / "test"
    / "images"
)


OUTPUT_JSON = (
    ROOT
    / "outputs"
    / "yolo_vit_predictions.json"
)


CLASS_FILE = (
    ROOT
    / "configs"
    / "classes.json"
)


ALIAS_FILE = (
    ROOT
    / "configs"
    / "aliases.json"
)



# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda"

YOLO_THRESHOLD = 0.70

VIT_THRESHOLD = 0.60



# ============================================================
# LOAD CLASSES
# ============================================================

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    CLASSES = json.load(f)


with open(ALIAS_FILE, "r", encoding="utf-8") as f:
    ALIASES = json.load(f)



def canonical_name(name):

    if name in ALIASES:
        return ALIASES[name]

    if name.startswith("BD_"):
        return name[3:]

    return name



# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO...")

yolo = YOLO(
    str(YOLO_MODEL)
)

print("YOLO loaded")


print("Loading ViT...")


vit = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=137
)


ckpt = torch.load(
    VIT_MODEL,
    map_location="cpu"
)


vit.load_state_dict(
    ckpt
)


vit.to(DEVICE)

vit.eval()


print("ViT loaded")



# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )
])



# ============================================================
# VIT PREDICT
# ============================================================

def vit_predict(crop):

    img = Image.fromarray(
        cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )
    )


    x = transform(
        img
    ).unsqueeze(0).to(DEVICE)


    with torch.no_grad():

        out = vit(x)

        prob = torch.softmax(
            out,
            dim=1
        )


    conf, idx = torch.max(
        prob,
        dim=1
    )


    name = CLASSES[
        str(idx.item())
    ]


    name = canonical_name(
        name
    )


    return (
        name,
        float(conf.item())
    )



# ============================================================
# EVALUATION
# ============================================================

results = []


images = list(
    IMAGE_DIR.glob("*")
)


print(
    f"Testing images: {len(images)}"
)



for img_path in tqdm(images):

    image = cv2.imread(
        str(img_path)
    )


    preds = []


    yolo_result = yolo.predict(

        source=image,

        conf=0.10,

        imgsz=640,

        device=0,

        verbose=False

    )[0]



    if yolo_result.boxes is None:
        continue



    for box in yolo_result.boxes:


        conf = float(
            box.conf[0]
        )


        cls = int(
            box.cls[0]
        )


        name = yolo.names[
            cls
        ]


        x1,y1,x2,y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )


        final_name = name

        source = "YOLO"


        final_conf = conf



        if conf < YOLO_THRESHOLD:


            crop = image[
                y1:y2,
                x1:x2
            ]


            if crop.size != 0:


                vit_name, vit_conf = vit_predict(
                    crop
                )


                if vit_conf >= VIT_THRESHOLD:

                    final_name = vit_name

                    final_conf = vit_conf

                    source = "ViT"



                else:

                    final_name = "unknown"

                    final_conf = vit_conf

                    source = "unknown"



        preds.append({

            "class": final_name,

            "confidence": final_conf,

            "source": source,

            "box":[

                int(x1),
                int(y1),
                int(x2),
                int(y2)

            ]

        })



    results.append({

        "image": img_path.name,

        "predictions": preds

    })



# ============================================================
# SAVE
# ============================================================

OUTPUT_JSON.parent.mkdir(
    exist_ok=True
)


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )



print("\nDONE")

print(
    "Saved:",
    OUTPUT_JSON
)