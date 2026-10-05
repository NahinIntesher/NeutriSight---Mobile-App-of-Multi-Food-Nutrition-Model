# src/train_nutrition.py

from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "datasets" / "nutrition5k"
IMAGE_ROOT = DATA_ROOT / "imagery" / "realsense_overhead"
NUTRITION_FILE = DATA_ROOT / "dish_nutrition_values.csv"
OUTPUT_DIR = ROOT / "checkpoints" / "nutrition5k_vit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LR = 3e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42

TARGETS = [
    "calories",
    "mass",
    "fat",
    "carb",
    "protein"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class NutritionDataset(Dataset):

    def __init__(self, dataframe, mean, std, train=False):

        self.df = dataframe.reset_index(drop=True)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )

        if train:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10
                ),
                transforms.ToTensor(),
                normalize
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),
                transforms.ToTensor(),
                normalize
            ])


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        dish_id = row["dish_id"]

        image_path = (
            IMAGE_ROOT
            / dish_id
            / "rgb.png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        target = torch.tensor(
            [
                float(row[x])
                for x in TARGETS
            ],
            dtype=torch.float32
        )

        target = (
            target - self.mean
        ) / self.std

        return image, target


# ============================================================
# SPLIT FILES
# ============================================================

def find_split_file(name):

    matches = list(
        DATA_ROOT.rglob(name)
    )

    return (
        matches[0]
        if matches
        else None
    )


TRAIN_SPLIT = find_split_file(
    "rgb_train_ids.txt"
)

TEST_SPLIT = find_split_file(
    "rgb_test_ids.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("NUTRITION5K ViT REGRESSION TRAINING")
print("=" * 80)

print(f"\nDevice: {DEVICE}")
print("Loading nutrition metadata...")

df = pd.read_csv(
    NUTRITION_FILE
)

required = [
    "dish_id",
    *TARGETS
]

missing = [
    x for x in required
    if x not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

df = (
    df[required]
    .dropna()
    .copy()
)

df["dish_id"] = (
    df["dish_id"]
    .astype(str)
)


# ============================================================
# AVAILABLE IMAGES
# ============================================================

print("Checking RGB images...")

available = [
    dish_id
    for dish_id in df["dish_id"]
    if (
        IMAGE_ROOT
        / dish_id
        / "rgb.png"
    ).exists()
]

df = (
    df[
        df["dish_id"].isin(available)
    ]
    .reset_index(drop=True)
)

print(
    f"Available dishes: {len(df)}"
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

if TRAIN_SPLIT and TEST_SPLIT:

    print(
        "\nUsing official Nutrition5k RGB split."
    )

    train_ids = {
        x.strip()
        for x in TRAIN_SPLIT.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    }

    test_ids = {
        x.strip()
        for x in TEST_SPLIT.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    }

    train_df = df[
        df["dish_id"].isin(train_ids)
    ].copy()

    test_df = df[
        df["dish_id"].isin(test_ids)
    ].copy()

    train_df = train_df.sample(
        frac=1.0,
        random_state=SEED
    ).reset_index(drop=True)

    val_size = max(
        1,
        int(len(train_df) * 0.10)
    )

    val_df = train_df.iloc[
        :val_size
    ].copy()

    train_df = train_df.iloc[
        val_size:
    ].copy()

else:

    print(
        "\nOfficial split files not found."
    )

    print(
        "Using deterministic 80/10/10 split."
    )

    df = df.sample(
        frac=1.0,
        random_state=SEED
    ).reset_index(drop=True)

    n = len(df)

    n_train = int(
        n * 0.80
    )

    n_val = int(
        n * 0.10
    )

    train_df = df.iloc[
        :n_train
    ].copy()

    val_df = df.iloc[
        n_train:n_train + n_val
    ].copy()

    test_df = df.iloc[
        n_train + n_val:
    ].copy()


print(
    f"\nTrain      : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


# ============================================================
# TARGET NORMALIZATION
# ============================================================

mean = (
    train_df[TARGETS]
    .mean()
    .values
)

std = (
    train_df[TARGETS]
    .std()
    .values
)

std = np.where(
    std < 1e-6,
    1.0,
    std
)

with open(
    OUTPUT_DIR / "target_stats.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "targets": TARGETS,
            "mean": mean.tolist(),
            "std": std.tolist()
        },
        f,
        indent=4
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    NutritionDataset(
        train_df,
        mean,
        std,
        train=True
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    NutritionDataset(
        val_df,
        mean,
        std,
        train=False
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    NutritionDataset(
        test_df,
        mean,
        std,
        train=False
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading ViT-Small...")

model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=True,
    num_classes=len(TARGETS)
).to(DEVICE)


# ============================================================
# TRAINING SETUP
# ============================================================

criterion = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = torch.cuda.amp.GradScaler(
    enabled=torch.cuda.is_available()
)

best_val = float("inf")
history = []


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Validation",
            leave=False
        )

        for images, targets in progress:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    targets
                )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            outputs = (
                outputs
                .detach()
                .cpu()
                .numpy()
            )

            targets = (
                targets
                .detach()
                .cpu()
                .numpy()
            )

            outputs = (
                outputs * std
                + mean
            )

            targets = (
                targets * std
                + mean
            )

            predictions.append(outputs)
            actuals.append(targets)

    predictions = np.concatenate(
        predictions
    )

    actuals = np.concatenate(
        actuals
    )

    mae = np.mean(
        np.abs(
            predictions - actuals
        ),
        axis=0
    )

    rmse = np.sqrt(
        np.mean(
            (
                predictions - actuals
            ) ** 2,
            axis=0
        )
    )

    overall_mae = float(
        np.mean(mae)
    )

    return (
        total_loss / len(loader.dataset),
        overall_mae,
        mae,
        rmse
    )


# ============================================================
# TRAINING
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch:02d}/{EPOCHS}",
        leave=True
    )

    for images, targets in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    scheduler.step()

    train_loss = (
        running_loss
        /
        len(train_loader.dataset)
    )

    (
        val_loss,
        val_mae,
        val_mae_each,
        val_rmse_each
    ) = evaluate(
        val_loader
    )

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val MAE: {val_mae:.2f}"
    )

    print(
        "MAE [cal,mass,fat,carb,protein]:",
        np.round(
            val_mae_each,
            2
        )
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_mae": val_mae
    })

    if val_mae < best_val:

        best_val = val_mae

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),
                "mean":
                    mean.tolist(),
                "std":
                    std.tolist(),
                "targets":
                    TARGETS,
                "epoch":
                    epoch,
                "val_mae":
                    float(val_mae)
            },
            OUTPUT_DIR
            / "best_model.pth"
        )

        print(
            "✓ Best model saved"
        )


# ============================================================
# TEST BEST MODEL
# ============================================================

print(
    "\nLoading best checkpoint..."
)

checkpoint = torch.load(
    OUTPUT_DIR / "best_model.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

(
    test_loss,
    test_mae,
    test_mae_each,
    test_rmse_each
) = evaluate(
    test_loader
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("NUTRITION5K TEST RESULTS")
print("=" * 80)

print(
    f"Test MAE: {test_mae:.2f}"
)

for name, mae_value, rmse_value in zip(
    TARGETS,
    test_mae_each,
    test_rmse_each
):

    print(
        f"{name:10s} | "
        f"MAE: {mae_value:.2f} | "
        f"RMSE: {rmse_value:.2f}"
    )

print(
    f"\nBest validation MAE: "
    f"{best_val:.2f}"
)

print(
    f"Checkpoint: "
    f"{OUTPUT_DIR / 'best_model.pth'}"
)


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    OUTPUT_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print("=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)