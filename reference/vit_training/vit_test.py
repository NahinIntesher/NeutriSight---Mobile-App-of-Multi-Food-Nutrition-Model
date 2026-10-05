import torch
import timm
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# নতুন ইম্পোর্টসমূহ (Metrics এবং Plotting এর জন্য)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():

    # ==========================
    # PATHS
    # ==========================
    MODEL_PATH = "models/best_model.pth"
    TEST_DIR = "datasets/final_dataset/test"

    # ==========================
    # DEVICE
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ==========================
    # DATASET
    # ==========================
    dataset = datasets.ImageFolder(TEST_DIR)

    num_classes = len(dataset.classes)
    class_names = dataset.classes  # ক্লাসগুলোর নাম (Heatmap এবং Report এর জন্য)

    # ==========================
    # MODEL
    # ==========================
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    # ==========================
    # TRANSFORM
    # ==========================
    data_config = timm.data.resolve_model_data_config(model)

    transform = timm.data.create_transform(
        **data_config,
        is_training=False
    )

    dataset.transform = transform

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,      # Windows-safe
        pin_memory=True
    )

    # ==========================
    # EVALUATION
    # ==========================
    all_preds = []
    all_labels = []
    confidence_sum = 0.0

    with torch.no_grad():

        for images, labels in tqdm(loader, desc="Testing"):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            confidence, preds = torch.max(probs, dim=1)

            # Metrics হিসাবের জন্য সব প্রেডিকশন ও আসল লেবেল লিস্টে জমা করছি
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            confidence_sum += confidence.sum().item()

    # array তে কনভার্ট করা হচ্ছে
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total = len(all_labels)

    # ==========================
    # METRICS GENERATION
    # ==========================
    # স্কিট-লার্ন দিয়ে ম্যাট্রিক্স এবং ক্লাসিফিকেশন রিপোর্ট তৈরি
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    correct = np.sum(all_preds == all_labels)
    accuracy = 100 * correct / total
    avg_confidence = 100 * confidence_sum / total

    # ==========================
    # PRINT RESULTS
    # ==========================
    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(f"Total Images      : {total}")
    print(f"Correct           : {correct}")
    print(f"Wrong             : {total - correct}")
    print(f"Overall Accuracy  : {accuracy:.2f}%")
    print(f"Average Confidence: {avg_confidence:.2f}%")
    print("==============================\n")
    
    print("CLASSIFICATION REPORT (Precision, Recall, F1-Score):")
    print("====================================================")
    print(report)
    print("====================================================\n")

    # ==========================
    # CONFUSION MATRIX HEATMAP
    # ==========================
    print("Generating Confusion Matrix Heatmap...")
    plt.figure(figsize=(10, 8))
    
    # Seaborn দিয়ে সুন্দর হিটম্যাপ তৈরি
    sns.heatmap(
        cm, 
        annot=True,          # বক্সের ভেতর সংখ্যা দেখাবে
        fmt="d",             # Integer ফরম্যাট
        cmap="Blues",        # নীল রঙের শেড
        xticklabels=class_names, 
        yticklabels=class_names
    )
    
    plt.title("Confusion Matrix Heatmap", fontsize=14, pad=15)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.tight_layout()
    
    # ইমেজ হিসেবে সেভ করবে এবং স্ক্রিনে শো করবে
    plt.savefig("confusion_matrix.png", dpi=300)
    print("Confusion Matrix saved as 'confusion_matrix.png'")
    plt.show()


if __name__ == "__main__":
    main()