import os

# Reduce Ultralytics console output
os.environ["YOLO_VERBOSE"] = "False"

from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


# ============================================================
# PATHS / SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = (
    ROOT
    / "datasets"
    / "merged_dataset"
    / "data.yaml"
)

RUN_NAME = "merged_yolo26n_seg"

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / RUN_NAME
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "weights"
    / "last.pt"
)


# ============================================================
# CUSTOM CLEAN PROGRESS BAR
# ============================================================

progress_bar = None


def get_losses(trainer):
    """
    Get running average losses from Ultralytics trainer.

    For YOLO segmentation:
    box_loss
    seg_loss
    cls_loss
    dfl_loss
    """

    try:
        losses = trainer.tloss

        if hasattr(losses, "detach"):
            losses = losses.detach().cpu().tolist()

        if not isinstance(losses, (list, tuple)):
            losses = [float(losses)]

        losses = [float(x) for x in losses]

        box_loss = losses[0] if len(losses) > 0 else 0.0
        seg_loss = losses[1] if len(losses) > 1 else 0.0
        cls_loss = losses[2] if len(losses) > 2 else 0.0
        dfl_loss = losses[3] if len(losses) > 3 else 0.0

        total_loss = (
            box_loss
            + seg_loss
            + cls_loss
            + dfl_loss
        )

        return (
            box_loss,
            seg_loss,
            cls_loss,
            dfl_loss,
            total_loss,
        )

    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


def on_train_epoch_start(trainer):

    global progress_bar

    total_batches = len(
        trainer.train_loader
    )

    progress_bar = tqdm(
        total=total_batches,
        desc=(
            f"Epoch "
            f"{trainer.epoch + 1}/"
            f"{trainer.args.epochs}"
        ),
        unit="batch",

        # IMPORTANT:
        # Keep completed epoch visible
        leave=True,

        dynamic_ncols=True,
        position=0,
        mininterval=0.5,
    )


def on_train_batch_end(trainer):

    global progress_bar

    if progress_bar is None:
        return

    progress_bar.update(1)

    (
        box_loss,
        seg_loss,
        cls_loss,
        dfl_loss,
        total_loss,
    ) = get_losses(trainer)

    progress_bar.set_postfix(
        {
            "loss": f"{total_loss:.4f}",
            "box": f"{box_loss:.4f}",
            "seg": f"{seg_loss:.4f}",
            "cls": f"{cls_loss:.4f}",
            "dfl": f"{dfl_loss:.4f}",
        },
        refresh=False,
    )


def on_train_epoch_end(trainer):

    global progress_bar

    if progress_bar is None:
        return

    (
        box_loss,
        seg_loss,
        cls_loss,
        dfl_loss,
        total_loss,
    ) = get_losses(trainer)

    progress_bar.n = progress_bar.total

    progress_bar.set_postfix(
        {
            "loss": f"{total_loss:.4f}",
            "box": f"{box_loss:.4f}",
            "seg": f"{seg_loss:.4f}",
            "cls": f"{cls_loss:.4f}",
            "dfl": f"{dfl_loss:.4f}",
        }
    )

    progress_bar.refresh()

    # Because leave=True,
    # completed epoch stays in terminal
    progress_bar.close()

    progress_bar = None


def on_train_end(trainer):

    global progress_bar

    if progress_bar is not None:
        progress_bar.close()
        progress_bar = None

    print()
    print("=" * 70)
    print("Training finished successfully.")
    print("=" * 70)


def add_callbacks(model):

    model.add_callback(
        "on_train_epoch_start",
        on_train_epoch_start
    )

    model.add_callback(
        "on_train_batch_end",
        on_train_batch_end
    )

    model.add_callback(
        "on_train_epoch_end",
        on_train_epoch_end
    )

    model.add_callback(
        "on_train_end",
        on_train_end
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("MULTI-FOOD SEGMENTATION TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Resume automatically
    # --------------------------------------------------------

    if LAST_CHECKPOINT.exists():

        print("\nCheckpoint found:")
        print(LAST_CHECKPOINT)

        print("\nResuming training...\n")

        model = YOLO(
            str(LAST_CHECKPOINT)
        )

        add_callbacks(model)

        model.train(
            resume=True
        )

        return


    # --------------------------------------------------------
    # Fresh training
    # --------------------------------------------------------

    print("\nNo checkpoint found.")
    print("Starting new training...\n")

    model = YOLO(
        "yolo26n-seg.pt"
    )

    add_callbacks(model)

    model.train(

        # Dataset
        data=str(DATA_YAML),

        # Training
        epochs=50,
        imgsz=640,

        # GTX 1660 Super 6 GB
        batch=4,
        device=0,

        # Windows + 16 GB RAM
        workers=0,

        # 1660 Super AMP compatibility issue
        amp=False,

        # Saving
        project=str(
            ROOT / "checkpoints"
        ),

        name=RUN_NAME,

        save=True,

        # Save every epoch
        save_period=1,

        # Stop if no improvement
        patience=12,

        # Generate graphs
        plots=True,

        # Reduce Ultralytics console spam
        verbose=False,

        exist_ok=True,
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()