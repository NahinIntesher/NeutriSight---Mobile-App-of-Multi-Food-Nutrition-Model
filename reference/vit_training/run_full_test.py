import cv2
import numpy as np
import joblib
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm  
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

IMAGE_PATH = "test1.jpg"
MODEL_PTH_PATH = "models/best_model.pth"

DOUBLE_LINE = "╚" + "═"*48 + "╝"
THICK_LINE  = "╔" + "═"*48 + "╗"
THIN_LINE   = "╟" + "─"*48 + "╢"

print("\n" + THICK_LINE)
print("║       INITIALIZING NUTRIENT AI PIPELINE        ║")
print(DOUBLE_LINE + "\n")

# -------------------
# 1. LOAD NUTRITION DB
# -------------------
print(" [➔] Loading Nutrition Database...", end="")
with open("nutrition_db.json", "r", encoding="utf-8") as f:
    nutrition_db = json.load(f)
print("\r [✔] Database Loaded Successfully.   ")

# -------------------
# 2. LOAD ViT MODEL
# -------------------
print(" [➔] Loading ViT Model (.pth)...", end="")
class_names = list(nutrition_db.keys())
MODEL_CLASSES = 137 

vit_model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=MODEL_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vit_model.load_state_dict(torch.load(MODEL_PTH_PATH, map_location=device, weights_only=True))
vit_model.to(device)
vit_model.eval()
print(f"\r [✔] ViT Model Loaded on {device.type.upper()}.      ")

# -------------------
# 3. LOAD SAM2 MODEL
# -------------------
print(" [➔] Loading SAM2 Model...", end="")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2 = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    "segmentation/checkpoints/sam2.1_hiera_tiny.pt",
    device=device
)
predictor = SAM2ImagePredictor(sam2)
print("\r [✔] SAM2 Model Loaded Successfully. ")

# -------------------
# 4. LOAD XGB]OOST MODELS
# -------------------
print(" [➔] Loading XGBoost Regression Models...", end="")
xgb_models = {
    "mass": joblib.load("models/xgboost/mass_xgb.pkl"),
    "calories": joblib.load("models/xgboost/calories_xgb.pkl"),
    "fat": joblib.load("models/xgboost/fat_xgb.pkl"),
    "carb": joblib.load("models/xgboost/carb_xgb.pkl"),
    "protein": joblib.load("models/xgboost/protein_xgb.pkl"),
}
print("\r [✔] All 5 XGBoost Models Loaded.         ")

print("\n" + "┌" + "─"*48 + "┐")
print("│          RUNNING IMAGE ANALYSIS PIPELINE       │")
print("└" + "─"*48 + "┘")

steps = ["ViT Food Classification", "SAM2 Image Segmentation", "XGBoost & DB Nutrient Inference"]

with tqdm(total=len(steps), desc="Processing", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
    
    # STEP 1
    pbar.set_postfix_str(steps[0])
    data_config = timm.data.resolve_model_data_config(vit_model)
    vit_transform = timm.data.create_transform(**data_config, is_training=False)
    
    pil_img = Image.open(IMAGE_PATH).convert("RGB")
    input_tensor = vit_transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = vit_model(input_tensor)
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
        
    food_name = class_names[predicted_class_idx].lower()
        
    time.sleep(0.4)
    pbar.update(1)
    
    # STEP 2
    pbar.set_postfix_str(steps[1])
    img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    predictor.set_image(img_rgb)
    box = np.array([int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)])
    masks, scores, _ = predictor.predict(box=box)
    mask = (masks[np.argmax(scores)] > 0.5).astype(np.uint8)
    
    mask_area = cv2.countNonZero(mask)
    x, y, bw, bh = cv2.boundingRect(mask)
    bbox_area = bw * bh
    
    features = np.array([[mask_area, bbox_area, w, h]])
    time.sleep(0.4)
    pbar.update(1)
    
    # STEP 3
    pbar.set_postfix_str(steps[2])
    xgb_pred = {}
    for k, m in xgb_models.items():
        xgb_pred[k] = float(m.predict(features)[0])
        
    detected_mass = max(0, xgb_pred["mass"])
    
    db_calc = {}
    
    if food_name.lower() in nutrition_db:
        db = nutrition_db[food_name.lower()]
        db_calc["calories"] = (db["calories"] / 100) * detected_mass
        db_calc["fat"] = (db["fat"] / 100) * detected_mass
        db_calc["carb"] = (db["carbs"] / 100) * detected_mass
        db_calc["protein"] = (db["protein"] / 100) * detected_mass
    else:
        db_calc = xgb_pred
        
    final_output = {}
    for k in ["calories", "fat", "carb", "protein"]:
        model_val = max(0, xgb_pred[k])
        db_val = max(0, db_calc[k])
        final_output[k] = (model_val + db_val) / 2
        
    time.sleep(0.4)
    pbar.update(1)
    pbar.set_postfix_str("Complete!")

# -------------------
# OUTPUT
# -------------------
cleaned_food_name = nutrition_db[food_name.lower()]["name"]

print("\n" + THICK_LINE)
print("║                 AI NUTRITION REPORT            ║")
print(THIN_LINE)
print(f"║ 🍽️  Detected Food : {cleaned_food_name:<26} ║")
print(f"║ ⚖️  Estimated Mass: {detected_mass:>7.2f} grams              ║")
print(THIN_LINE)
print("║    NUTRITIONAL BREAKDOWN (Fused AI + DB)       ║")
print(THIN_LINE)
print(f"║ 🔥 Calories      : {final_output['calories']:>7.2f} kcal                ║")
print(f"║ 🍞 Carbohydrates : {final_output['carb']:>7.2f} g                   ║")
print(f"║ 🥩 Protein       : {final_output['protein']:>7.2f} g                   ║")
print(f"║ 🥑 Fat           : {final_output['fat']:>7.2f} g                   ║")
print(DOUBLE_LINE + "\n")