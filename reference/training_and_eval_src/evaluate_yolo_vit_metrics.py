# src/evaluate_yolo_vit_metrics.py

from pathlib import Path
import json
import cv2



# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent


PRED_JSON = (
    ROOT
    / "outputs"
    / "yolo_vit_predictions.json"
)


LABEL_DIR = (
    ROOT
    / "datasets"
    / "merged_dataset"
    / "test"
    / "labels"
)


IMAGE_DIR = (
    ROOT
    / "datasets"
    / "merged_dataset"
    / "test"
    / "images"
)


YOLO_CLASS_FILE = (
    ROOT
    / "configs"
    / "yolo_classes.json"
)



# ============================================================
# SETTINGS
# ============================================================

IOU_THRESHOLD = 0.5



# ============================================================
# LOAD CLASS MAP
# ============================================================

with open(
    YOLO_CLASS_FILE,
    "r",
    encoding="utf-8"
) as f:

    classes = json.load(f)



CLASS_TO_ID = {
    v:int(k)
    for k,v in classes.items()
}



# ============================================================
# LOAD PREDICTIONS
# ============================================================

with open(
    PRED_JSON,
    "r",
    encoding="utf-8"
) as f:

    predictions = json.load(f)



# ============================================================
# IOU
# ============================================================

def iou(box1,box2):

    x1=max(box1[0],box2[0])
    y1=max(box1[1],box2[1])

    x2=min(box1[2],box2[2])
    y2=min(box1[3],box2[3])


    inter=max(0,x2-x1)*max(0,y2-y1)


    area1=(box1[2]-box1[0])*(box1[3]-box1[1])

    area2=(box2[2]-box2[0])*(box2[3]-box2[1])


    union=area1+area2-inter


    return inter/union if union>0 else 0



# ============================================================
# NORMALIZE BOX
# ============================================================

def normalize_box(box,w,h):

    if max(box)<=1:

        return [

            box[0]*w,
            box[1]*h,
            box[2]*w,
            box[3]*h

        ]

    return box



# ============================================================
# METRICS
# ============================================================

TP=0
FP=0
FN=0


total_gt=0



for item in predictions:


    image_name=item["image"]


    img=cv2.imread(
        str(
            IMAGE_DIR/image_name
        )
    )


    if img is None:
        continue


    h,w=img.shape[:2]


    label_file=(
        LABEL_DIR/
        (
            Path(image_name).stem
            +
            ".txt"
        )
    )


    if not label_file.exists():
        continue



    # -----------------------------
    # Ground Truth
    # -----------------------------

    gt=[]


    with open(label_file,"r") as f:

        for line in f:


            data=line.strip().split()


            if not data:
                continue


            cls=int(data[0])


            pts=list(
                map(
                    float,
                    data[1:]
                )
            )


            xs=[]
            ys=[]


            for i in range(0,len(pts),2):

                xs.append(
                    pts[i]*w
                )

                ys.append(
                    pts[i+1]*h
                )


            gt.append({

                "class":cls,

                "box":[

                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys)

                ],

                "used":False

            })


    total_gt += len(gt)



    # -----------------------------
    # Predictions
    # -----------------------------

    preds=[]


    for p in item["predictions"]:


        name=p["class"]


        if name not in CLASS_TO_ID:
            continue


        preds.append({

            "class":
            CLASS_TO_ID[name],

            "box":
            normalize_box(
                p["box"],
                w,
                h
            ),

            "conf":
            p.get(
                "confidence",
                1.0
            )

        })



    preds=sorted(

        preds,

        key=lambda x:x["conf"],

        reverse=True

    )



    # -----------------------------
    # Matching
    # -----------------------------

    for pred in preds:


        matched=False


        for g in gt:


            if g["used"]:
                continue



            if (

                pred["class"]
                ==
                g["class"]

                and

                iou(
                    pred["box"],
                    g["box"]
                )
                >=
                IOU_THRESHOLD

            ):

                TP+=1

                g["used"]=True

                matched=True

                break



        if not matched:

            FP+=1



    for g in gt:

        if not g["used"]:

            FN+=1



# ============================================================
# FINAL RESULTS
# ============================================================

precision = TP/(TP+FP+1e-9)

recall = TP/(TP+FN+1e-9)


f1 = (
    2*precision*recall
    /
    (precision+recall+1e-9)
)


accuracy = (
    TP
    /
    (TP+FP+FN+1e-9)
)



print("="*80)

print("YOLO + ViT FINAL METRICS")

print("="*80)


print(
    "True Positive:",
    TP
)

print(
    "False Positive:",
    FP
)

print(
    "False Negative:",
    FN
)

print()


print(
    "Precision:",
    round(precision,4)
)


print(
    "Recall:",
    round(recall,4)
)


print(
    "F1 Score:",
    round(f1,4)
)


print(
    "Accuracy:",
    round(accuracy,4)
)


print("="*80)

print("NOTE:")
print("mAP50 and mAP50-95 require confidence-ranked COCO evaluation.")
print("This script calculates detection metrics only.")

print("="*80)