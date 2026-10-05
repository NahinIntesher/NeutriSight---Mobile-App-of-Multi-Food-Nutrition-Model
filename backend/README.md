# NutriSight Backend v3

FastAPI backend for the NutriSight mobile app.

## Folder layout

Keep these folders as siblings:

```text
NutriSight/
├─ backend/
├─ frontend/
└─ models/
   ├─ yolo/best.pt
   ├─ vit_lstm/best_model.pth
   └─ nutrition/best_model.pth
```

The model files are deliberately **not included** in this backend package. Reuse your existing `models` folder.

## Model flow

1. YOLO performs food detection/segmentation.
2. Detections with confidence below `0.70` are sent to the supplied 137-class ViT classifier.
3. ViT predictions at or above `0.60` replace the uncertain YOLO label. Otherwise the item is marked `unknown`.
4. Nutrition5k ViT regression estimates calories, mass, fat, carbs and protein for the **whole meal image**.

`configs/classes.json` contains the supplied class order. The folder name `vit_lstm` is preserved for compatibility, but the supplied checkpoint is loaded as the ViT-Small architecture used by the training/test code.

## Windows setup

Run once:

```bat
setup.bat
```

Then start the API:

```bat
start.bat
```

API: `http://127.0.0.1:8000`
Docs: `http://127.0.0.1:8000/docs`
Health: `http://127.0.0.1:8000/health`

## Expo upload fix

The app now uses `POST /api/analyze/raw` for native image uploads. This avoids the Expo `Unsupported FormDataPart implementation` problem. The old multipart `POST /api/analyze` endpoint remains for compatibility/testing.

## Account flow

New accounts start with an empty nutrition profile. After the first sign-in the frontend opens the Meal Insights onboarding form. Users may save it or skip it. Skip state is saved so the form does not reopen every login.

## Existing account/history data

The SQLite file is stored in `backend/data/nutrisight.sqlite3`. If you already have real accounts/history, keep a backup of that file before replacing folders.

## Useful commands

```bat
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python verify_model_contract.py
```

`verify_model_contract.py` checks the sibling model folder, class count and nutrition checkpoint contract without performing inference.

## Notes

- Images are processed in memory and are not written to the backend by the analysis endpoint.
- Saved users get private scan results/history in SQLite.
- Guest scans are not saved.
- Dietary alerts are conservative name-based checks only.
- Nutrition output is an image-based estimate, not medical advice.
