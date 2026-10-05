# yolo_vit_test.py

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



# ============================================================
# LOAD CONFIG
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
print("YOLO FIRST + ViT FALLBACK v2")
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



def calculate_iou(a,b):

    x1 = max(
        a["x1"],
        b["x1"]
    )

    y1 = max(
        a["y1"],
        b["y1"]
    )

    x2 = min(
        a["x2"],
        b["x2"]
    )

    y2 = min(
        a["y2"],
        b["y2"]
    )


    intersection = max(
        0,
        x2-x1
    ) * max(
        0,
        y2-y1
    )


    area_a = (
        a["x2"]-a["x1"]
    ) * (
        a["y2"]-a["y1"]
    )


    area_b = (
        b["x2"]-b["x1"]
    ) * (
        b["y2"]-b["y1"]
    )


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    if union == 0:
        return 0


    return intersection / union



def nms_merge(detections):

    detections = sorted(
        detections,
        key=lambda x:x["conf"],
        reverse=True
    )


    keep = []


    while detections:

        best = detections.pop(0)

        keep.append(best)


        remaining = []


        for d in detections:

            if calculate_iou(
                best,
                d
            ) < NMS_IOU_THRESHOLD:

                remaining.append(d)


        detections = remaining


    return keep

# ============================================================
# ViT CLASSIFICATION
# ============================================================

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
# YOLO INFERENCE
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
# COLLECT DETECTIONS
# ============================================================

detections = []


if result.boxes is not None:


    for box in result.boxes:


        conf = float(
            box.conf[0]
        )


        cls = int(
            box.cls[0]
        )


        x1,y1,x2,y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )


        detections.append({

            "name":
            yolo.names[cls],

            "conf":
            conf,

            "x1":
            x1,

            "y1":
            y1,

            "x2":
            x2,

            "y2":
            y2
        })



# Remove duplicate overlapping boxes

detections = nms_merge(
    detections
)



# ============================================================
# FINAL DECISION
# ============================================================

print("\n")
print("="*80)
print("FINAL PREDICTIONS")
print("="*80)



if len(detections) == 0:

    print("No food detected")


else:


    accepted = 0


    for i, det in enumerate(
        detections,
        start=1
    ):


        yolo_name = det["name"]

        yolo_conf = det["conf"]


        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]



        final_name = yolo_name

        source = "YOLO"

        final_conf = yolo_conf



        print("\n")
        print("-"*70)

        print(
            f"Region {i}"
        )


        print(
            "YOLO:",
            yolo_name,
            f"{yolo_conf*100:.1f}%"
        )



        # ====================================================
        # YOLO WEAK -> ViT FALLBACK
        # ====================================================


        if yolo_conf < YOLO_CONF_THRESHOLD:


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

                    final_conf = vit_conf

                    source = "ViT fallback"



                else:


                    final_name = "unknown"

                    final_conf = vit_conf

                    source = "Rejected"



        print(
            "FINAL:",
            final_name
        )


        print(
            "CONFIDENCE:",
            f"{final_conf*100:.1f}%"
        )


        print(
            "SOURCE:",
            source
        )



        # ====================================================
        # DRAW OUTPUT
        # ====================================================


        color = (
            0,
            255,
            0
        )


        cv2.rectangle(

            image,

            (x1,y1),

            (x2,y2),

            color,

            2
        )


        cv2.putText(

            image,

            final_name,

            (x1,y1-10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            color,

            2
        )


        accepted += 1



# ============================================================
# SAVE IMAGE
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
    "Processed regions:",
    len(detections)
)


print(
    "Saved:",
    OUTPUT_PATH
)