import json
import logging
import os
from pathlib import Path
from threading import Lock

LOGGER = logging.getLogger("nutrisight.pipeline")


class FoodPipeline:
    """Serialized inference worker for the local NutriSight model stack."""

    YOLO_ACCEPT = 0.70
    VIT_ACCEPT = 0.60
    NMS_IOU = 0.50

    def __init__(self):
        self.root = Path(
            os.getenv(
                "MODELS_DIR",
                str(Path(__file__).resolve().parents[2] / "models"),
            )
        )
        self.lock = Lock()
        self.loaded = False
        self.error = None
        self.fallback = None
        self.fallback_note = "ViT refinement has not been loaded yet."

    def status(self):
        return {
            "loaded": self.loaded,
            "device": getattr(self, "device", "auto"),
            "error": self.error,
            "yolo_present": (self.root / "yolo/best.pt").is_file(),
            "nutrition_present": (self.root / "nutrition/best_model.pth").is_file(),
            "classifier_present": (self.root / "vit_lstm/best_model.pth").is_file(),
            "fallback_enabled": self.fallback is not None,
            "fallback_note": self.fallback_note,
            "thresholds": {
                "yolo_accept": self.YOLO_ACCEPT,
                "vit_accept": self.VIT_ACCEPT,
                "nms_iou": self.NMS_IOU,
            },
            "preprocessing": {
                "classifier": "224x224 + ImageNet normalization, matching supplied ViT test code",
                "nutrition": "224x224 + ImageNet normalization; checkpoint target mean/std restored",
            },
        }

    def load(self):
        import timm
        import torch
        from torchvision import transforms
        from ultralytics import YOLO

        self.torch = torch
        requested = os.getenv("DEVICE", "").strip().lower()
        if requested:
            self.device = requested
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        yolo_path = self.root / "yolo/best.pt"
        nutrition_path = self.root / "nutrition/best_model.pth"
        classifier_path = self.root / "vit_lstm/best_model.pth"
        labels_path = Path(__file__).resolve().parents[1] / "configs/classes.json"

        missing = [str(p) for p in (yolo_path, nutrition_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing required model file(s): " + ", ".join(missing))

        self.yolo = YOLO(str(yolo_path))

        nutrition_ckpt = torch.load(nutrition_path, map_location="cpu", weights_only=True)
        self.targets = list(nutrition_ckpt["targets"])
        if set(self.targets) != {"calories", "mass", "fat", "carb", "protein"}:
            raise ValueError(f"Unexpected nutrition target order: {self.targets}")
        self.nutrition = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=len(self.targets),
        )
        self.nutrition.load_state_dict(nutrition_ckpt["model_state_dict"], strict=True)
        self.nutrition.to(self.device).eval()
        self.mean = torch.tensor(nutrition_ckpt["mean"], dtype=torch.float32, device=self.device)
        self.std = torch.tensor(nutrition_ckpt["std"], dtype=torch.float32, device=self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.fallback = None
        if classifier_path.is_file() and labels_path.is_file():
            try:
                labels = json.loads(labels_path.read_text(encoding="utf-8"))
                weights = torch.load(classifier_path, map_location="cpu", weights_only=True)
                weights = weights.get("model_state_dict", weights)
                head = weights.get("head.weight")
                if not isinstance(labels, list) or not all(isinstance(x, str) and x.strip() for x in labels):
                    raise ValueError("classes.json must be a non-empty JSON string list")
                if head is None or int(head.shape[0]) != len(labels):
                    raise ValueError(
                        f"Classifier outputs ({None if head is None else int(head.shape[0])}) do not match class mapping ({len(labels)})"
                    )
                model = timm.create_model(
                    "vit_small_patch16_224",
                    pretrained=False,
                    num_classes=len(labels),
                )
                model.load_state_dict(weights, strict=True)
                model.to(self.device).eval()
                self.labels = labels
                self.fallback = model
                self.classifier_transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225],
                        ),
                    ]
                )
                self.fallback_note = (
                    f"ViT refinement ready with {len(labels)} classes. "
                    f"YOLO < {self.YOLO_ACCEPT:.2f} is refined; ViT >= {self.VIT_ACCEPT:.2f} is accepted."
                )
            except Exception as exc:
                self.fallback_note = f"ViT refinement disabled: {exc}"
                LOGGER.exception("Classifier load failed")
        else:
            self.fallback_note = "ViT refinement disabled because classifier weights or classes.json are missing."

        self.loaded = True
        self.error = None

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(1e-6, area_a + area_b - inter)

    @staticmethod
    def _clean_label(label):
        return label[3:] if label.startswith("BD_") else label

    def analyze(self, image):
        with self.lock:
            try:
                if not self.loaded:
                    self.load()
                return self._predict(image)
            except Exception as exc:
                self.error = str(exc)
                raise

    def _predict(self, image):
        torch = self.torch
        with torch.inference_mode():
            yolo_result = self.yolo.predict(
                image,
                imgsz=640,
                conf=0.10,
                retina_masks=True,
                max_det=50,
                device=(0 if str(self.device).startswith("cuda") else self.device),
                verbose=False,
            )[0]

            indices = list(range(len(yolo_result.boxes))) if yolo_result.boxes is not None else []
            indices.sort(
                key=lambda i: float(yolo_result.boxes[i].conf.item()),
                reverse=True,
            )
            kept_boxes = []
            foods = []

            for i in indices:
                box = yolo_result.boxes[i]
                xy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                if any(self._iou(xy, previous) >= self.NMS_IOU for previous in kept_boxes):
                    continue
                kept_boxes.append(xy)

                yolo_name = str(yolo_result.names[int(box.cls.item())])
                yolo_conf = float(box.conf.item())
                final_name = yolo_name
                final_conf = yolo_conf
                source = "YOLO"
                suggestion = None

                if yolo_conf < self.YOLO_ACCEPT and self.fallback is not None:
                    x1 = max(0, min(image.width - 1, int(round(xy[0]))))
                    y1 = max(0, min(image.height - 1, int(round(xy[1]))))
                    x2 = max(x1 + 1, min(image.width, int(round(xy[2]))))
                    y2 = max(y1 + 1, min(image.height, int(round(xy[3]))))
                    crop = image.crop((x1, y1, x2, y2))
                    logits = self.fallback(
                        self.classifier_transform(crop).unsqueeze(0).to(self.device)
                    )
                    probabilities = logits.softmax(-1)[0]
                    vit_score, vit_idx = probabilities.max(0)
                    vit_conf = float(vit_score.item())
                    vit_name = self.labels[int(vit_idx.item())]
                    suggestion = {
                        "name": self._clean_label(vit_name),
                        "confidence": round(vit_conf, 4),
                    }
                    if vit_conf >= self.VIT_ACCEPT:
                        final_name = vit_name
                        final_conf = vit_conf
                        source = "ViT fallback"
                    else:
                        final_name = "unknown"
                        final_conf = max(yolo_conf, vit_conf)
                        source = "Uncertain"

                polygon = []
                if yolo_result.masks is not None and i < len(yolo_result.masks.xyn):
                    polygon = yolo_result.masks.xyn[i].tolist()

                foods.append(
                    {
                        "name": self._clean_label(final_name),
                        "confidence": round(final_conf, 4),
                        "source": source,
                        "yolo": {
                            "name": self._clean_label(yolo_name),
                            "confidence": round(yolo_conf, 4),
                        },
                        "suggestion": suggestion,
                        "bbox": [
                            xy[0] / image.width,
                            xy[1] / image.height,
                            xy[2] / image.width,
                            xy[3] / image.height,
                        ],
                        "polygon": polygon,
                    }
                )

            nutrition = None
            if foods:
                values = (
                    self.nutrition(self.transform(image).unsqueeze(0).to(self.device))[0]
                    * self.std
                    + self.mean
                ).clamp(min=0)
                if not torch.isfinite(values).all():
                    raise ValueError("Non-finite nutrition output")
                nutrition = {
                    ("carbs" if key == "carb" else key): round(float(value), 1)
                    for key, value in zip(self.targets, values)
                }

            warnings = []
            if not foods:
                warnings.append("No food was detected. Retake the photo in good lighting with the complete meal visible.")
            else:
                warnings.append(
                    "Nutrition is an image-based estimate for the entire meal, not a nutrition value for each detected crop."
                )
                warnings.append(
                    "Hidden ingredients, recipes and exact portion weights cannot be verified from a photo."
                )
            if self.fallback is None:
                warnings.append(self.fallback_note)

            return {
                "foods": foods,
                "nutrition": nutrition,
                "nutrition_scope": "whole_meal",
                "image_size": [image.width, image.height],
                "warnings": warnings,
                "models": {
                    "detector": "YOLO segmentation",
                    "nutrition": "Nutrition5k ViT-Small regression",
                    "refinement": self.fallback is not None,
                },
            }
