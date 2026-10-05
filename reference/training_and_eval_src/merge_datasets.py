from pathlib import Path
from collections import defaultdict
import json
import os
import re
import shutil
import csv

import numpy as np
import yaml
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FOODBD_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "FoodBD"
    / "root"
    / "FoodBD"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "merged_dataset"
)

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# DishSeg image folder candidates
DISH_IMAGE_CANDIDATES = [
    PROJECT_ROOT / "datasets" / "DishSeg24k_images" / "images",
]

# DishSeg annotation folder candidates
DISH_ANNOTATION_CANDIDATES = [
    PROJECT_ROOT / "datasets" / "DishSeg24K" / "annotations",
]


# ============================================================
# OPTIONAL CLASS ALIASES
# ============================================================

# Do NOT aggressively merge classes yet.
# Add verified aliases here later if necessary.

MANUAL_ALIASES = {
    # Example:
    # "dal": "daal",
    # "chapati": "ruti",
}


# ============================================================
# HELPERS
# ============================================================

def find_existing(candidates, description):
    for path in candidates:
        if path.exists():
            print(f"{description}: {path}")
            return path

    raise FileNotFoundError(
        f"\nCould not find {description}.\n"
        f"Checked:\n" +
        "\n".join(str(x) for x in candidates)
    )


def normalize_class_name(name):
    name = str(name).strip().lower()

    name = name.replace("_", "-")
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")

    if name in MANUAL_ALIASES:
        name = MANUAL_ALIASES[name]

    return name


def polygon_area(poly):
    points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * abs(
        np.dot(x, np.roll(y, 1))
        - np.dot(y, np.roll(x, 1))
    )


def choose_largest_polygon(segmentation):
    if not isinstance(segmentation, list):
        return None

    valid = []

    for poly in segmentation:
        if isinstance(poly, list) and len(poly) >= 6:
            valid.append(poly)

    if not valid:
        return None

    return max(valid, key=polygon_area)


def link_or_copy(src, dst):
    """
    Try hard-link first.
    This saves huge disk space on the same Windows drive.
    If hard-link fails, normal copy is used.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ============================================================
# LOAD FOODBD
# ============================================================

def load_foodbd():

    print("\n" + "=" * 70)
    print("READING FOODBD")
    print("=" * 70)

    yaml_path = FOODBD_ROOT / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config["names"]

    if isinstance(names, dict):
        names = [
            names[i] if i in names else names[str(i)]
            for i in range(len(names))
        ]

    names = [normalize_class_name(x) for x in names]

    records = []

    for split in ["train", "valid", "test"]:

        image_dir = FOODBD_ROOT / split / "images"
        label_dir = FOODBD_ROOT / split / "labels"

        image_files = []

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_files.extend(image_dir.glob(ext))

        image_files = [
            x for x in image_files
            if not x.name.startswith("._")
        ]

        print(f"{split}: {len(image_files)} images")

        for image_path in tqdm(
            image_files,
            desc=f"FoodBD {split}"
        ):

            label_path = label_dir / f"{image_path.stem}.txt"

            annotations = []
            classes = set()

            if label_path.exists():

                with open(
                    label_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    for line in f:

                        parts = line.strip().split()

                        if len(parts) < 7:
                            continue

                        class_id = int(float(parts[0]))

                        if class_id >= len(names):
                            continue

                        coords = [
                            float(x)
                            for x in parts[1:]
                        ]

                        if len(coords) % 2 != 0:
                            continue

                        class_name = names[class_id]

                        annotations.append(
                            {
                                "class": class_name,
                                "coords": coords,
                            }
                        )

                        classes.add(class_name)

            records.append(
                {
                    "source": "foodbd",
                    "image": image_path,
                    "annotations": annotations,
                    "classes": classes,
                }
            )

    print(
        f"\nFoodBD total loaded: "
        f"{len(records)} images"
    )

    return records


# ============================================================
# LOAD DISHSEG24K
# ============================================================

def load_dishseg():

    print("\n" + "=" * 70)
    print("READING DISHSEG24K")
    print("=" * 70)

    image_root = find_existing(
        DISH_IMAGE_CANDIDATES,
        "DishSeg24K image folder"
    )

    annotation_root = find_existing(
        DISH_ANNOTATION_CANDIDATES,
        "DishSeg24K annotation folder"
    )

    json_files = sorted(
        annotation_root.glob("instances_*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No instances_*.json found inside "
            f"{annotation_root}"
        )

    print("Annotation files:")

    for x in json_files:
        print(" ", x.name)

    records = []

    skipped_missing = 0
    skipped_rle = 0
    skipped_invalid = 0

    for json_path in json_files:

        print(f"\nReading {json_path.name}")

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:
            coco = json.load(f)

        categories = {
            category["id"]:
            normalize_class_name(category["name"])
            for category in coco["categories"]
        }

        annotations_by_image = defaultdict(list)

        for ann in coco["annotations"]:
            annotations_by_image[
                ann["image_id"]
            ].append(ann)

        for image_info in tqdm(
            coco["images"],
            desc=json_path.stem
        ):

            file_name = image_info["file_name"]

            if Path(file_name).name.startswith("._"):
                continue

            image_path = image_root / file_name

            if not image_path.exists():
                image_path = (
                    image_root
                    / Path(file_name).name
                )

            if not image_path.exists():
                skipped_missing += 1
                continue

            width = image_info["width"]
            height = image_info["height"]

            yolo_annotations = []
            classes = set()

            image_annotations = (
                annotations_by_image.get(
                    image_info["id"],
                    []
                )
            )

            for ann in image_annotations:

                category_id = ann["category_id"]

                if category_id not in categories:
                    continue

                segmentation = ann.get(
                    "segmentation"
                )

                # RLE mask
                if isinstance(segmentation, dict):
                    skipped_rle += 1
                    continue

                polygon = choose_largest_polygon(
                    segmentation
                )

                if polygon is None:
                    skipped_invalid += 1
                    continue

                coords = []

                for i in range(
                    0,
                    len(polygon),
                    2
                ):

                    x = polygon[i] / width
                    y = polygon[i + 1] / height

                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))

                    coords.extend([x, y])

                if len(coords) < 6:
                    continue

                class_name = categories[
                    category_id
                ]

                yolo_annotations.append(
                    {
                        "class": class_name,
                        "coords": coords,
                    }
                )

                classes.add(class_name)

            records.append(
                {
                    "source": "dishseg",
                    "image": image_path,
                    "annotations": yolo_annotations,
                    "classes": classes,
                }
            )

    print(
        f"\nDishSeg24K loaded: "
        f"{len(records)} images"
    )

    print(
        f"Missing images skipped: "
        f"{skipped_missing}"
    )

    print(
        f"RLE annotations skipped: "
        f"{skipped_rle}"
    )

    print(
        f"Invalid polygons skipped: "
        f"{skipped_invalid}"
    )

    return records


# ============================================================
# MULTILABEL STRATIFIED SPLIT
# ============================================================

def create_splits(records, class_to_id):

    print("\n" + "=" * 70)
    print("CREATING 80 / 10 / 10 STRATIFIED SPLIT")
    print("=" * 70)

    y = np.zeros(
        (
            len(records),
            len(class_to_id)
        ),
        dtype=np.uint8
    )

    for i, record in enumerate(records):

        for class_name in record["classes"]:

            y[
                i,
                class_to_id[class_name]
            ] = 1

    X = np.arange(
        len(records)
    ).reshape(-1, 1)

    first_split = (
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=SEED
        )
    )

    train_idx, temp_idx = next(
        first_split.split(X, y)
    )

    X_temp = X[temp_idx]
    y_temp = y[temp_idx]

    second_split = (
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=SEED
        )
    )

    val_local, test_local = next(
        second_split.split(
            X_temp,
            y_temp
        )
    )

    val_idx = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


# ============================================================
# WRITE DATASET
# ============================================================

def write_dataset(
    records,
    splits,
    class_to_id
):

    print("\n" + "=" * 70)
    print("WRITING MERGED DATASET")
    print("=" * 70)

    if OUTPUT_ROOT.exists():

        print(
            f"Removing old generated dataset:\n"
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    for split in [
        "train",
        "val",
        "test"
    ]:

        (
            OUTPUT_ROOT
            / split
            / "images"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_ROOT
            / split
            / "labels"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    split_summary = []

    for split_name, indices in splits.items():

        print(
            f"\nWriting {split_name}: "
            f"{len(indices)} images"
        )

        for number, index in enumerate(
            tqdm(indices)
        ):

            record = records[index]

            src = record["image"]

            extension = src.suffix.lower()

            # Unique filename
            filename = (
                f"{record['source']}_"
                f"{index:06d}_"
                f"{src.stem}"
                f"{extension}"
            )

            image_dst = (
                OUTPUT_ROOT
                / split_name
                / "images"
                / filename
            )

            label_dst = (
                OUTPUT_ROOT
                / split_name
                / "labels"
                / (
                    Path(filename).stem
                    + ".txt"
                )
            )

            link_or_copy(
                src,
                image_dst
            )

            lines = []

            for ann in record[
                "annotations"
            ]:

                class_id = class_to_id[
                    ann["class"]
                ]

                coords = " ".join(
                    f"{x:.8f}"
                    for x in ann["coords"]
                )

                lines.append(
                    f"{class_id} {coords}"
                )

            with open(
                label_dst,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(lines)
                )

        split_summary.append(
            {
                "split": split_name,
                "images": len(indices)
            }
        )

    return split_summary


# ============================================================
# REPORTS
# ============================================================

def write_reports(
    records,
    splits,
    classes
):

    print("\nGenerating reports...")

    class_report = (
        OUTPUT_ROOT
        / "class_mapping.csv"
    )

    sources_by_class = defaultdict(set)

    for record in records:
        for cls in record["classes"]:
            sources_by_class[cls].add(
                record["source"]
            )

    with open(
        class_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "class_id",
                "class_name",
                "sources"
            ]
        )

        for class_id, name in enumerate(
            classes
        ):

            writer.writerow(
                [
                    class_id,
                    name,
                    ",".join(
                        sorted(
                            sources_by_class[
                                name
                            ]
                        )
                    )
                ]
            )

    distribution_file = (
        OUTPUT_ROOT
        / "class_distribution.csv"
    )

    with open(
        distribution_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "class",
                "train_images",
                "val_images",
                "test_images",
            ]
        )

        counts = {}

        for class_name in classes:

            counts[class_name] = {
                "train": 0,
                "val": 0,
                "test": 0,
            }

        for split_name, indices in splits.items():

            for index in indices:

                for cls in records[
                    index
                ]["classes"]:

                    counts[cls][
                        split_name
                    ] += 1

        for class_name in classes:

            writer.writerow(
                [
                    class_name,
                    counts[class_name][
                        "train"
                    ],
                    counts[class_name][
                        "val"
                    ],
                    counts[class_name][
                        "test"
                    ],
                ]
            )


# ============================================================
# DATA.YAML
# ============================================================

def write_yaml(classes):

    yaml_path = (
        OUTPUT_ROOT
        / "data.yaml"
    )

    config = {
        "path": str(
            OUTPUT_ROOT.resolve()
        ),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": {
            i: name
            for i, name in enumerate(
                classes
            )
        },
    }

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("FOODBD + DISHSEG24K MERGER")
    print("=" * 70)

    foodbd_records = load_foodbd()
    dishseg_records = load_dishseg()

    records = (
        foodbd_records
        + dishseg_records
    )

    print(
        f"\nCombined images: "
        f"{len(records)}"
    )

    classes = sorted(
        {
            cls
            for record in records
            for cls in record["classes"]
        }
    )

    class_to_id = {
        name: i
        for i, name in enumerate(
            classes
        )
    }

    print(
        f"Unified classes: "
        f"{len(classes)}"
    )

    splits = create_splits(
        records,
        class_to_id
    )

    print("\nSplit result:")

    print(
        "Train:",
        len(splits["train"])
    )

    print(
        "Validation:",
        len(splits["val"])
    )

    print(
        "Test:",
        len(splits["test"])
    )

    write_dataset(
        records,
        splits,
        class_to_id
    )

    write_reports(
        records,
        splits,
        classes
    )

    write_yaml(
        classes
    )

    print("\n")
    print("=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)

    print(
        f"\nDataset:\n{OUTPUT_ROOT}"
    )

    print(
        f"\nClasses: "
        f"{len(classes)}"
    )

    print("\nGenerated:")

    print("  data.yaml")
    print("  class_mapping.csv")
    print("  class_distribution.csv")

    print(
        "\nNext step: validate merged "
        "dataset before training."
    )


if __name__ == "__main__":
    main()