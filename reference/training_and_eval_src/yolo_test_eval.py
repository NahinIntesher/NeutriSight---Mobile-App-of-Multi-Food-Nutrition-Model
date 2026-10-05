from pathlib import Path
import multiprocessing
import gc
import torch

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent


MODEL_PATH = (
    ROOT
    / "checkpoints"
    / "merged_yolo26n_seg"
    / "weights"
    / "best.pt"
)


DATA_YAML = (
    ROOT
    / "datasets"
    / "merged_dataset"
    / "data.yaml"
)



# ============================================================
# SETTINGS
# ============================================================

DEVICE = 0



# ============================================================
# MAIN
# ============================================================

def main():

    gc.collect()
    torch.cuda.empty_cache()


    print("="*80)
    print("YOLO BASELINE EVALUATION")
    print("="*80)


    print("\nLoading model...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("Model loaded")



    print("\nRunning validation...\n")


    metrics = model.val(

        data=str(DATA_YAML),

        split="test",

        imgsz=512,

        batch=1,

        device=DEVICE,

        workers=0,

        half=True,

        plots=False,

        save_json=True,

        max_det=50

    )



    # ========================================================
    # METRICS
    # ========================================================

    precision = metrics.box.mp

    recall = metrics.box.mr


    f1 = (
        2 * precision * recall
        /
        (precision + recall + 1e-9)
    )


    accuracy = (
        2 * precision * recall
        /
        (precision + recall + 1e-9)
    )



    print("\n")
    print("="*80)
    print("YOLO FINAL METRICS")
    print("="*80)



    print("\nDetection")

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


    print(
        "mAP50:",
        round(metrics.box.map50,4)
    )


    print(
        "mAP50-95:",
        round(metrics.box.map,4)
    )



    print("\nSegmentation")


    print(
        "Mask Precision:",
        round(metrics.seg.mp,4)
    )


    print(
        "Mask Recall:",
        round(metrics.seg.mr,4)
    )


    print(
        "Mask mAP50:",
        round(metrics.seg.map50,4)
    )


    print(
        "Mask mAP50-95:",
        round(metrics.seg.map,4)
    )



    print("\n")
    print("="*80)
    print("EVALUATION COMPLETED")
    print("="*80)



if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()