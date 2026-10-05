from pathlib import Path
import cv2
import numpy as np

ROOT = Path("datasets/FoodBD/root/FoodBD")

IMAGE = ROOT / "train/images/FoodBD-0002.jpg"
LABEL = ROOT / "train/labels/FoodBD-0002.txt"

names = [
    'achar','apple','badam','banana','beef','begun-vaji','beguni',
    'biriyani','bread','burinda','cabbage','carrot','cauliflower',
    'cha','chichinga-vaji','chicken','chola','chomchom','chop-alu',
    'cucumber','daal','dates','dundol','egg','egg-boiled',
    'egg-poached','fish','grapes','green-pea','guava','jilapi',
    'juice','kabab','khichuri','kochu','korola-vaji','kumra-vaji',
    'lal-shak','lau','lemon','mango','milk','noodles','okra',
    'orange','papaya','payesh','peyaju','pineapple','piyaju',
    'potato','potato-mashed','potol','puffed-rice','puri','rice',
    'roll','ruti','salad','shak','shim','sweets','tomato','vaji',
    'vegetable','vorta','watermelon'
]

image = cv2.imread(str(IMAGE))
h, w = image.shape[:2]

overlay = image.copy()

with open(LABEL, "r") as f:
    for line in f:
        values = list(map(float, line.split()))

        class_id = int(values[0])
        coords = values[1:]

        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.polylines(image, [points], True, (0, 255, 0), 2)

        x, y, bw, bh = cv2.boundingRect(points)

        cv2.rectangle(
            image,
            (x, y),
            (x + bw, y + bh),
            (255, 0, 0),
            2
        )

        cv2.putText(
            image,
            names[class_id],
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

result = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

Path("outputs").mkdir(exist_ok=True)

cv2.imwrite(
    "outputs/foodbd_annotation_check.jpg",
    result
)

print("Saved: outputs/foodbd_annotation_check.jpg")