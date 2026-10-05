"""Quick model-file contract check without running inference."""
import json
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
MODELS = Path(os.getenv('MODELS_DIR', str(ROOT.parent / 'models')))
CLASSES = ROOT / 'configs' / 'classes.json'

labels = json.loads(CLASSES.read_text(encoding='utf-8'))
vit_path = MODELS / 'vit_lstm' / 'best_model.pth'
nut_path = MODELS / 'nutrition' / 'best_model.pth'
yolo_path = MODELS / 'yolo' / 'best.pt'

for path in (yolo_path, vit_path, nut_path):
    if not path.is_file():
        raise SystemExit(f'MISSING: {path}')

vit = torch.load(vit_path, map_location='cpu', weights_only=True)
vit = vit.get('model_state_dict', vit)
head = vit.get('head.weight')
if head is None or head.shape[0] != len(labels):
    raise SystemExit(f'ViT/classes mismatch: head={None if head is None else tuple(head.shape)}, labels={len(labels)}')

nut = torch.load(nut_path, map_location='cpu', weights_only=True)
expected = {'calories', 'mass', 'fat', 'carb', 'protein'}
if set(nut.get('targets', [])) != expected:
    raise SystemExit(f'Unexpected nutrition targets: {nut.get("targets")}')

print('OK: YOLO checkpoint found')
print(f'OK: ViT classifier outputs = {head.shape[0]} classes')
print(f'OK: classes.json entries = {len(labels)}')
print(f'OK: nutrition targets = {nut["targets"]}')
print('Model file contract is compatible with NutriSight backend v3.')
