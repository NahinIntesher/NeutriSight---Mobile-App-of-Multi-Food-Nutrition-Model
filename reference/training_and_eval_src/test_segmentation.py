from pathlib import Path
import csv
import numpy as np

from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# PROJECT SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

# Current final merged segmentation model
MODEL_PATH = (
    ROOT
    / "checkpoints"
    / "merged_yolo26n_seg"
    / "weights"
    / "best.pt"
)

# Merged dataset YAML
DATA_YAML = (
    ROOT
    / "datasets"
    / "merged_dataset"
    / "data.yaml"
)

# Merged test images
TEST_IMAGES = (
    ROOT
    / "datasets"
    / "merged_dataset"
    / "test"
    / "images"
)

# Main output folder
OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "final_segmentation_evaluation"
)

VAL_NAME = "metrics"
PRED_NAME = "qualitative_masks"

VAL_OUTPUT = OUTPUT_ROOT / VAL_NAME
PRED_OUTPUT = OUTPUT_ROOT / PRED_NAME

PER_CLASS_CSV = (
    OUTPUT_ROOT
    / "per_class_segmentation_metrics.csv"
)

OVERALL_CSV = (
    OUTPUT_ROOT
    / "overall_segmentation_metrics.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_files():

    print("=" * 75)
    print("FINAL SEGMENTATION MODEL EVALUATION")
    print("=" * 75)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\nDataset YAML not found:\n{DATA_YAML}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"\nTest image folder not found:\n{TEST_IMAGES}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nModel:")
    print(MODEL_PATH)

    print("\nDataset:")
    print(DATA_YAML)

    print("\nTest images:")
    print(TEST_IMAGES)


def get_class_name(names, class_id):

    if isinstance(names, dict):
        return names.get(
            int(class_id),
            str(class_id)
        )

    try:
        return names[int(class_id)]
    except Exception:
        return str(class_id)


def to_float(value):

    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================
# OVERALL METRICS
# ============================================================

def save_overall_metrics(metrics):

    print("\n")
    print("=" * 75)
    print("OVERALL TEST RESULTS")
    print("=" * 75)

    rows = []

    # --------------------------------------------------------
    # Bounding-box metrics
    # --------------------------------------------------------

    if hasattr(metrics, "box"):

        box_precision = to_float(
            metrics.box.mp
        )

        box_recall = to_float(
            metrics.box.mr
        )

        box_map50 = to_float(
            metrics.box.map50
        )

        box_map5095 = to_float(
            metrics.box.map
        )

        print("\nBounding Box")
        print("-" * 40)

        print(
            f"Precision       : "
            f"{box_precision:.4f}"
        )

        print(
            f"Recall          : "
            f"{box_recall:.4f}"
        )

        print(
            f"Box mAP@50      : "
            f"{box_map50:.4f}"
        )

        print(
            f"Box mAP@50-95   : "
            f"{box_map5095:.4f}"
        )

        rows.extend([
            ["Box Precision", box_precision],
            ["Box Recall", box_recall],
            ["Box mAP@50", box_map50],
            ["Box mAP@50-95", box_map5095],
        ])

    # --------------------------------------------------------
    # Segmentation metrics
    # --------------------------------------------------------

    if hasattr(metrics, "seg"):

        mask_precision = to_float(
            metrics.seg.mp
        )

        mask_recall = to_float(
            metrics.seg.mr
        )

        mask_map50 = to_float(
            metrics.seg.map50
        )

        mask_map5095 = to_float(
            metrics.seg.map
        )

        print("\nSegmentation Masks")
        print("-" * 40)

        print(
            f"Precision       : "
            f"{mask_precision:.4f}"
        )

        print(
            f"Recall          : "
            f"{mask_recall:.4f}"
        )

        print(
            f"Mask mAP@50     : "
            f"{mask_map50:.4f}"
        )

        print(
            f"Mask mAP@50-95  : "
            f"{mask_map5095:.4f}"
        )

        rows.extend([
            ["Mask Precision", mask_precision],
            ["Mask Recall", mask_recall],
            ["Mask mAP@50", mask_map50],
            ["Mask mAP@50-95", mask_map5095],
        ])

    # Save overall results
    with open(
        OVERALL_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Metric",
            "Value"
        ])

        writer.writerows(rows)

    print("\nOverall metrics saved:")
    print(OVERALL_CSV)


# ============================================================
# PER-CLASS SEGMENTATION METRICS
# ============================================================

def save_per_class_metrics(metrics, names):

    print("\n")
    print("=" * 75)
    print("PER-CLASS SEGMENTATION RESULTS")
    print("=" * 75)

    if not hasattr(metrics, "seg"):
        print(
            "Segmentation metrics were not found."
        )
        return

    seg = metrics.seg

    try:
        class_indices = np.array(
            seg.ap_class_index
        ).astype(int)

        precision = np.array(seg.p)
        recall = np.array(seg.r)
        ap50 = np.array(seg.ap50)
        ap5095 = np.array(seg.ap)

    except Exception as e:

        print(
            "Could not extract per-class metrics:"
        )

        print(e)

        return

    rows = []

    print(
        "\n"
        f"{'ID':<5}"
        f"{'Class':<30}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'AP50':>12}"
        f"{'AP50-95':>12}"
    )

    print("-" * 85)

    for i, class_id in enumerate(
        class_indices
    ):

        class_name = get_class_name(
            names,
            class_id
        )

        p = (
            to_float(precision[i])
            if i < len(precision)
            else 0.0
        )

        r = (
            to_float(recall[i])
            if i < len(recall)
            else 0.0
        )

        a50 = (
            to_float(ap50[i])
            if i < len(ap50)
            else 0.0
        )

        a5095 = (
            to_float(ap5095[i])
            if i < len(ap5095)
            else 0.0
        )

        print(
            f"{class_id:<5}"
            f"{class_name:<30}"
            f"{p:>12.4f}"
            f"{r:>12.4f}"
            f"{a50:>12.4f}"
            f"{a5095:>12.4f}"
        )

        rows.append([
            class_id,
            class_name,
            p,
            r,
            a50,
            a5095
        ])

    # Sort CSV by AP50-95, best first
    rows_sorted = sorted(
        rows,
        key=lambda x: x[5],
        reverse=True
    )

    with open(
        PER_CLASS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "class_name",
            "precision",
            "recall",
            "mask_AP50",
            "mask_AP50_95"
        ])

        writer.writerows(
            rows_sorted
        )

    print("\nPer-class metrics saved:")
    print(PER_CLASS_CSV)

    # --------------------------------------------------------
    # Show best classes
    # --------------------------------------------------------

    print("\nTop 10 classes by Mask AP50-95")
    print("-" * 55)

    for row in rows_sorted[:10]:

        print(
            f"{row[1]:<30} "
            f"{row[5]:.4f}"
        )

    # --------------------------------------------------------
    # Show weakest classes
    # --------------------------------------------------------

    print("\nWeakest 10 classes by Mask AP50-95")
    print("-" * 55)

    for row in rows_sorted[-10:]:

        print(
            f"{row[1]:<30} "
            f"{row[5]:.4f}"
        )


# ============================================================
# NUMERICAL EVALUATION
# ============================================================

def run_validation(model):

    print("\n")
    print("=" * 75)
    print("RUNNING TEST-SET EVALUATION")
    print("=" * 75)

    metrics = model.val(

        # Dataset
        data=str(DATA_YAML),

        # Evaluate TEST split
        split="test",

        # Same image resolution as training
        imgsz=640,

        # GTX 1660 Super
        batch=4,
        device=0,

        # Windows stability
        workers=0,

        # Generate:
        # confusion matrix
        # PR curve
        # F1 curve
        # P curve
        # R curve
        plots=True,

        # Output folder
        project=str(OUTPUT_ROOT),
        name=VAL_NAME,

        exist_ok=True,

        verbose=True
    )

    return metrics


# ============================================================
# QUALITATIVE MASK PREDICTIONS
# ============================================================

def generate_predictions(model):

    print("\n")
    print("=" * 75)
    print("GENERATING QUALITATIVE MASK RESULTS")
    print("=" * 75)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    total_images = sum(
        1
        for file in TEST_IMAGES.iterdir()
        if file.suffix.lower() in image_extensions
    )

    print(f"\nTotal test images: {total_images}\n")

    prediction_stream = model.predict(
        source=str(TEST_IMAGES),

        conf=0.25,
        imgsz=640,
        device=0,

        save=True,
        save_txt=True,
        save_conf=True,
        retina_masks=True,

        project=str(OUTPUT_ROOT),
        name=PRED_NAME,

        exist_ok=True,

        # Keep YOLO console clean
        verbose=False,

        # Process one result at a time
        stream=True
    )

    count = 0

    progress_bar = tqdm(
        prediction_stream,
        total=total_images,
        desc="Generating predictions",
        unit="image",
        dynamic_ncols=True
    )

    for _ in progress_bar:
        count += 1

    print(
        f"\nQualitative predictions generated "
        f"for {count} images."
    )

    print("\nSaved to:")
    print(PRED_OUTPUT)

    print(
        f"\nQualitative predictions generated "
        f"for {count} images."
    )

    print("\nSaved to:")
    print(PRED_OUTPUT)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Check everything exists
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # 2. Load final model
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("LOADING BEST MODEL")
    print("=" * 75)

    model = YOLO(
        str(MODEL_PATH)
    )

    print("\nModel loaded successfully.")

    print(
        f"Number of classes: "
        f"{len(model.names)}"
    )

    # --------------------------------------------------------
    # 3. Evaluate numerically
    # --------------------------------------------------------

    metrics = run_validation(
        model
    )

    # --------------------------------------------------------
    # 4. Overall metrics
    # --------------------------------------------------------

    save_overall_metrics(
        metrics
    )

    # --------------------------------------------------------
    # 5. Per-class metrics
    # --------------------------------------------------------

    save_per_class_metrics(
        metrics,
        model.names
    )

    # --------------------------------------------------------
    # 6. Save qualitative examples
    # --------------------------------------------------------

    generate_predictions(
        model
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("FINAL EVALUATION COMPLETED")
    print("=" * 75)

    print("\nMain output folder:")
    print(OUTPUT_ROOT)

    print("\nOverall metrics:")
    print(OVERALL_CSV)

    print("\nPer-class metrics:")
    print(PER_CLASS_CSV)

    print("\nEvaluation plots:")
    print(VAL_OUTPUT)

    print("\nQualitative masks:")
    print(PRED_OUTPUT)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()