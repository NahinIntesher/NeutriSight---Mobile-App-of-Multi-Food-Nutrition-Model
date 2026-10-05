import json
import time
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

# PyTorch ফিউচার ওয়ার্নিং স্ক্রিন থেকে হাইড করার জন্য
warnings.filterwarnings("ignore", category=FutureWarning)

def print_progress(message: str):
    current_time = time.strftime("%H:%M:%S")
    print(f"[{current_time}] ⏳ {message}...", flush=True)
    time.sleep(0.2)

print_progress("Booting up NeutriPlus Dual-Model Engine (ViT + YOLOv8)")

try:
    import torch
    import timm
    from PIL import Image
    from ultralytics import YOLO  # YOLOv8 আবার যুক্ত করা হলো
    print("[SUCCESS] Core frameworks and YOLOv8 loaded successfully.\n")
except ModuleNotFoundError as e:
    print(f"\n[CRITICAL ERROR] Missing library: {e}")
    print("Please run: pip install timm torch pillow ultralytics")
    sys.exit(1)


class FoodNutritionPipeline:
    def __init__(self, model_path: str, db_path: str, base_model_name: str = "vit_small_patch16_224"):
        self.db_path = Path(db_path)
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Pipeline anchored to device: {self.device.type.upper()}\n")
        
        self.nutrition_db = self._load_database()
        
        # --- 💡 CRITICAL SIZE MISMATCH FIX 💡 ---
        # চেকপয়েন্টের হেড শেপ (137) এর সাথে মেলাতে মডেলটিকে অবশ্যই ১৩৭ ক্লাস দিয়ে ইনিশিয়ালাইজ করতে হবে
        self.num_classes = 137  
        
        self.train_config, self.model = self._load_model()
        self.class_labels = list(self.nutrition_db.keys())
        
        # --- YOLOv8 Segmentation Model Initialization ---
        print_progress("Loading Pretrained YOLOv8-Segmentation Model")
        self.yolo_model = YOLO("yolov8n-seg.pt")

    def _load_database(self) -> Dict[str, Any]:
        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k.lower().strip(): v for k, v in data.items()}

    def _load_model(self) -> tuple[Any, Any]:
        print_progress("Initializing Vision Transformer Architecture")
        model = timm.create_model(self.base_model_name, pretrained=False, num_classes=self.num_classes)
        
        print_progress("Loading Weights from best_model.pth")
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(checkpoint)
        
        data_config = timm.data.resolve_model_data_config(model)
        processor = timm.data.create_transform(**data_config, is_training=False)
        
        model.to(self.device)
        model.eval()
        print("[SUCCESS] ViT Model weights coupled perfectly without shape mismatch.\n")
        return processor, model

    def estimate_weight_via_yolo(self, image_path: str) -> float:
        """
        Uses YOLOv8-seg to count food pixels and estimate weight.
        """
        print_progress("YOLOv8: Segmenting food boundaries to calculate volume area")
        results = self.yolo_model(image_path, verbose=False)
        
        if results and results[0].masks is not None:
            pixel_area = int(torch.sum(results[0].masks.data[0]).item())
            print(f"[INFO] YOLOv8 detected food mask occupying {pixel_area} pixels.")
            
            pixels_per_gram = 25 
            estimated_weight = pixel_area / pixels_per_gram
            return round(estimated_weight, 1)
        
        print("[WARNING] YOLOv8 could not isolate food mask. Falling back to default standard.")
        return 100.0

    def analyze_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        img_path = Path(image_path)
        if not img_path.exists(): 
            print(f"[ERROR] Image path does not exist: {image_path}")
            return None

        try:
            # ১. ওয়ান-শট ওয়ান ট্র্যান্সফর্ম ইনফারেন্স (ViT)
            with Image.open(img_path) as img:
                image = img.convert("RGB")
            inputs = self.train_config(image).unsqueeze(0).to(self.device)
            
            print_progress("ViT: Processing neural vision loops")
            with torch.no_grad():
                outputs = self.model(inputs)
                predicted_class_idx = outputs.argmax(-1).item()
            
            # ২. আউটপুট লেবেল বাউন্ডারি সেফগার্ড চেক
            if predicted_class_idx < len(self.class_labels):
                clean_label = self.class_labels[predicted_class_idx]
            else:
                print(f"[WARNING] Predicted index {predicted_class_idx} is out of current JSON database bounds.")
                return None
            
            # ৩. ওজনের হিসাব (YOLOv8)
            detected_weight = self.estimate_weight_via_yolo(image_path)
            ratio = detected_weight / 100.0
            
            # ৪. ডেটাবেজ ক্রস-রেফারেন্স
            nutrition_data = self.nutrition_db.get(clean_label)
            
            if nutrition_data:
                calculated_nutrition = {
                    "calories": round(nutrition_data["calories"] * ratio, 1),
                    "protein": round(nutrition_data["protein"] * ratio, 1),
                    "carbs": round(nutrition_data["carbs"] * ratio, 1),
                    "fat": round(nutrition_data["fat"] * ratio, 1),
                }
            else:
                calculated_nutrition = None

            return {
                "food_name": clean_label.replace("bd_", "").replace("_", " ").title(),
                "weight_g": detected_weight,
                "nutrition": calculated_nutrition
            }

        except Exception as e:
            print(f"[CRITICAL] Error during image analysis: {e}")
            return None


# --- Execution Entry Point ---
if __name__ == "__main__":
    MODEL_IDENTIFIER = "models/best_model.pth"  
    DATABASE_PATH = "nutrition_db.json"
    TEST_IMAGE_PATH = "test2.jpg"  # আপনার টেস্ট ইমেজ দিন এখানে

    try:
        pipeline = FoodNutritionPipeline(model_path=MODEL_IDENTIFIER, db_path=DATABASE_PATH)
        result = pipeline.analyze_image(image_path=TEST_IMAGE_PATH)
        
        if result and result["nutrition"]:
            nutr = result["nutrition"]
            print("\n" + "="*45)
            print(f" 📊 FINAL NUTRITION PROFILE ({result['weight_g']}g): {result['food_name']}")
            print("="*45)
            print(f"  • Calories : {nutr['calories']} kcal")
            print(f"  • Protein  : {nutr['protein']}g")
            print(f"  • Carbs    : {nutr['carbs']}g")
            print(f"  • Fat      : {nutr['fat']}g")
            print("="*45 + "\n")
        else:
            print("\n[NOTICE] Inference succeeded but item profile could not be fetched.")
            
    except Exception as general_error:
        print(f"\n[CRITICAL Failure]: {general_error}")