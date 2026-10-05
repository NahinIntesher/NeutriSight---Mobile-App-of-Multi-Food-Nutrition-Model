# yolo_vit_test_v1.py

from pathlib import Path
import json
import cv2
import torch
import timm

from ultralytics import YOLO
from torchvision import transforms
from PIL import Image


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


IMAGE_PATH = (
    ROOT
    / "src"
    / "3.jpg"
)


OUTPUT_PATH = (
    ROOT
    / "src"
    / "yolo_vit_output.jpg"
)


# Config files

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

TAXONOMY_FILE = (
    ROOT
    / "configs"
    / "taxonomy.json"
)



# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_CONF_THRESHOLD = 0.70

VIT_CONF_THRESHOLD = 0.60

NMS_IOU_THRESHOLD = 0.50

IMG_SIZE = 224



# ============================================================
# LOAD JSON FILES
# ============================================================

with open(
    CLASS_FILE,
    "r",
    encoding="utf-8"
) as f:
    CLASS_NAMES = json.load(f)


with open(
    ALIAS_FILE,
    "r",
    encoding="utf-8"
) as f:
    ALIASES = json.load(f)


with open(
    TAXONOMY_FILE,
    "r",
    encoding="utf-8"
) as f:
    TAXONOMY = json.load(f)



# ============================================================
# LOAD MODELS
# ============================================================

print("="*80)
print("YOLO FIRST + ViT FALLBACK")
print("="*80)


print("\nLoading YOLO...")

yolo = YOLO(
    str(YOLO_MODEL)
)

print("YOLO loaded")



print("\nLoading ViT...")


vit = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=137
)


checkpoint = torch.load(
    VIT_MODEL,
    map_location="cpu"
)


vit.load_state_dict(
    checkpoint
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
# HELPERS
# ============================================================

def canonical_name(name):

    if name in ALIASES:
        return ALIASES[name]

    if name.startswith("BD_"):
        return name[3:]

    return name


def apply_nms(boxes, iou_threshold=0.5):

    if len(boxes) == 0:
        return []


    keep = []

    boxes = sorted(
        boxes,
        key=lambda x: x["conf"],
        reverse=True
    )


    while boxes:

        current = boxes.pop(0)

        keep.append(current)


        filtered = []

        for b in boxes:

            x1 = max(
                current["x1"],
                b["x1"]
            )

            y1 = max(
                current["y1"],
                b["y1"]
            )

            x2 = min(
                current["x2"],
                b["x2"]
            )

            y2 = min(
                current["y2"],
                b["y2"]
            )


            inter = max(0,x2-x1) * max(0,y2-y1)


            area1 = (
                current["x2"]-current["x1"]
            ) * (
                current["y2"]-current["y1"]
            )


            area2 = (
                b["x2"]-b["x1"]
            ) * (
                b["y2"]-b["y1"]
            )


            union = area1 + area2 - inter


            iou = inter / union if union else 0


            if iou < iou_threshold:
                filtered.append(b)


        boxes = filtered


    return keep


def classify_vit(crop):

    image = Image.fromarray(
        cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )
    )


    x = transform(
        image
    ).unsqueeze(0)


    x = x.to(DEVICE)


    with torch.no_grad():

        output = vit(x)

        prob = torch.softmax(
            output,
            dim=1
        )


    conf, idx = torch.max(
        prob,
        dim=1
    )


    name = CLASS_NAMES[
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
# RUN YOLO
# ============================================================

print("\nRunning YOLO...")


results = yolo.predict(

    source=str(IMAGE_PATH),

    conf=0.10,

    imgsz=640,

    device=0,

    retina_masks=True,

    verbose=False
)


result = results[0]


image = cv2.imread(
    str(IMAGE_PATH)
)



# ============================================================
# DECISION
# ============================================================

print("\n")
print("="*80)
print("FINAL PREDICTIONS")
print("="*80)


detections = []


for box in result.boxes:

    conf = float(box.conf[0])

    cls = int(box.cls[0])

    x1,y1,x2,y2 = (
        box.xyxy[0]
        .cpu()
        .numpy()
        .astype(int)
    )


    detections.append({

        "conf": conf,

        "class": yolo.names[cls],

        "x1": x1,

        "y1": y1,

        "x2": x2,

        "y2": y2
    })


detections = apply_nms(
    detections,
    NMS_IOU_THRESHOLD
)


if result.boxes is None:

    print("No food detected")

else:


    for i, det in enumerate(
        detections,
        start=1
    ):

        conf = det["conf"]
        yolo_name = det["class"]

        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]

        print("\nRegion",i)

        print(
            "YOLO:",
            yolo_name,
            f"{conf*100:.1f}%"
        )


        final_name = yolo_name
        source = "YOLO"


        # LOW CONFIDENCE -> ViT

        if conf < YOLO_CONF_THRESHOLD:


            crop = image[
                y1:y2,
                x1:x2
            ]


            if crop.size != 0:

                vit_name, vit_conf = classify_vit(
                    crop
                )

                print(
                    "ViT:",
                    vit_name,
                    f"{vit_conf*100:.1f}%"
                )

                if vit_conf >= VIT_CONF_THRESHOLD:
                    final_name = vit_name
                    source = "ViT fallback"
                else:
                    final_name = "unknown"
                    source = "Rejected"
                    
        print(
            "FINAL:",
            final_name
        )

        print(
            "SOURCE:",
            source
        )



        # draw

        cv2.rectangle(
            image,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        cv2.putText(
            image,
            final_name,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )



# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    str(OUTPUT_PATH),
    image
)


print("\n")
print("="*80)
print("COMPLETED")
print("="*80)

print(
    "Saved:",
    OUTPUT_PATH
)