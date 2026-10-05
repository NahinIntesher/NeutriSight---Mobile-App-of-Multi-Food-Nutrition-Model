import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

uec_path = "datasets/UECFOOD256"
bd_path = "datasets/BDFood"
output = "datasets/yolo_dataset"

img_train = f"{output}/images/train"
img_val = f"{output}/images/val"
lbl_train = f"{output}/labels/train"
lbl_val = f"{output}/labels/val"

for p in [img_train, img_val, lbl_train, lbl_val]:
    os.makedirs(p, exist_ok=True)

class_map = {}
class_id = 0


def process(base_path, prefix):

    global class_id

    classes = os.listdir(base_path)

    for cls in tqdm(classes, desc=base_path):

        cls_path = os.path.join(base_path, cls)

        if not os.path.isdir(cls_path):
            continue

        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        if len(images) == 0:
            continue

        if cls not in class_map:
            class_map[cls] = class_id
            class_id += 1

        cid = class_map[cls]

        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        def save(img_list, img_dir, lbl_dir):

            for img in tqdm(img_list, leave=False):

                src = os.path.join(cls_path, img)

                new_name = prefix + cls + "_" + img

                shutil.copy(src, os.path.join(img_dir, new_name))

                label_path = os.path.join(lbl_dir, new_name.replace(".jpg", ".txt"))

                with open(label_path, "w") as f:
                    f.write(f"{cid} 0.5 0.5 1.0 1.0")

        save(train_imgs, img_train, lbl_train)
        save(val_imgs, img_val, lbl_val)


process(uec_path, "UEC_")
process(bd_path, "BD_")

# Save class names
classes_file = os.path.join(output, "classes.txt")

with open(classes_file, "w", encoding="utf-8") as f:
    for cls, idx in sorted(class_map.items(), key=lambda x: x[1]):
        f.write(cls + "\n")

# Create data.yaml automatically
yaml_path = "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {output}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"nc: {len(class_map)}\n")
    f.write("names:\n")

    for cls, idx in sorted(class_map.items(), key=lambda x: x[1]):
        f.write(f"  - {cls}\n")

print(f"Saved classes -> {classes_file}")
print(f"Saved yaml -> {yaml_path}")


print("DONE")
print("Total classes:", len(class_map))