import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image, ImageFile
import timm
from tqdm import tqdm

# Fix truncated image errors in PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ====================================
# AUTOMATIC DATASET CLEANUP
# ====================================
def auto_clean_directory(target_directory):
    """Scans and deletes corrupted images with real-time progress updates."""
    if not os.path.exists(target_directory):
        return
    print(f"🧹 Scanning and cleaning directory: {target_directory}")
    valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    corrupted_count = 0
    checked_count = 0

    total_files = sum(1 for root, _, files in os.walk(target_directory)
                      for f in files if f.lower().endswith(valid_extensions))

    print(f"📊 Found total {total_files} images to verify. Processing...")

    for root, _, files in os.walk(target_directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                checked_count += 1

                if checked_count % 10000 == 0 or checked_count == total_files:
                    print(f"   ⏳ Progress: [{checked_count}/{total_files}] images checked...")

                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    with Image.open(file_path) as img:
                        img.load()
                except Exception:
                    print(f"\n   ❌ Removing corrupted image: {file_path}")
                    try:
                        os.remove(file_path)
                        corrupted_count += 1
                    except OSError:
                        pass

    if corrupted_count > 0:
        print(f"✅ Cleaned {corrupted_count} corrupted images from {target_directory}.\n")
    else:
        print(f"✨ No corrupted images found in {target_directory}.\n")


# ====================================
# MAIN EXECUTION FUNCTION
# ====================================
def main():
    # ====================================
    # DEVICE SETUP
    # ====================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    torch.backends.cudnn.benchmark = True
    use_amp = (device.type == "cuda")

    # ====================================
    # PATHS
    # ====================================
    train_dir = "C:/final_dataset/train"
    val_dir = "C:/final_dataset/val"

    model_save_path = "models"
    os.makedirs(model_save_path, exist_ok=True)

    # Run cleanup before building datasets
    auto_clean_directory(train_dir)
    auto_clean_directory(val_dir)

    # ====================================
    # INITIALIZE MODEL FIRST
    # ====================================
    num_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    print(f"Detected {num_classes} classes. Initializing ViT...")
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=num_classes
    )
    model = model.to(device)

    # ====================================
    # TRANSFORMS (Using optimized timm configs)
    # ====================================
    data_config = timm.data.resolve_model_data_config(model)
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    # ====================================
    # DATASETS & DATALOADERS
    # ====================================
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)


    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # ====================================
    # LOSS, OPTIMIZER, & SCHEDULER
    # ====================================
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)

    # ===== FIX 1: AMP SAFETY (NO CRASH) =====
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    
    # ====================================
    # CHECKPOINT RESUME SYSTEM
    # ====================================
    checkpoint_path = os.path.join(model_save_path, "checkpoint.pth")
    start_epoch = 0
    best_val_acc = 0.0
    counter = 0

    if os.path.exists(checkpoint_path):
        print("🔄 Resuming from checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])

        best_val_acc = checkpoint['best_val_acc']
        counter = checkpoint['counter']
        start_epoch = checkpoint['epoch'] + 1

        print(f"✅ Resumed from after Epoch {start_epoch}")

    # ====================================
    # TRAIN SETTINGS
    # ====================================
    num_epochs = 37
    patience = 7
    

    # ====================================
    # TRAIN LOOP
    # ====================================
    for epoch in range(start_epoch, num_epochs):

        print(f"\n🚀 Starting Epoch [{epoch+1}/{num_epochs}]...")
        print("-" * 40)

        # ------------------------
        # TRAINING PHASE
        # ------------------------
        model.train()
        train_loss = 0.0
        successful_train_batches = 0

        pbar = tqdm(train_loader)
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # ===== FIX 2: SAFE AUTOCAST =====
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                successful_train_batches += 1

                pbar.set_description(f"Epoch {epoch+1} | 🏋️ Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
                # if batch_idx % 100 == 0:
                #     tqdm.write(f"   🏋️ Training Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"\n❌ Train Batch Error {batch_idx}: {e}")
                continue

        avg_train_loss = train_loss / max(successful_train_batches, 1)

        # ------------------------
        # VALIDATION PHASE
        # ------------------------
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        successful_val_batches = 0

        with torch.no_grad():
            vbar = tqdm(val_loader)
            for batch_idx, (images, labels) in enumerate(vbar):
                try:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    # ===== FIX 3: SAFE AUTOCAST =====
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    successful_val_batches += 1

                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    vbar.set_description(f"Epoch {epoch+1} | 🔍 Validation Batch [{batch_idx+1}/{len(val_loader)}] | Loss: {loss.item():.4f}")

                    # if batch_idx % 100 == 0:
                    #     tqdm.write(f"   🔍 Validation Batch [{batch_idx+1}/{len(val_loader)}] | Loss: {loss.item():.4f}")

                except Exception as e:
                    print(f"\n❌ Val Batch Error {batch_idx}: {e}")
                    continue

        avg_val_loss = val_loss / max(successful_val_batches, 1)
        val_acc = correct / max(total, 1)

        # ------------------------
        # PRINT PERFORMANCE
        # ------------------------
        print("\n" + "="*40)
        print(f"📊 Results - Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss  : {avg_val_loss:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")
        print("="*40 + "\n")

        # ------------------------
        # SAVE BEST MODEL
        # ------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0

            best_model_path = os.path.join(model_save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New Best Model Saved! Accuracy: {best_val_acc:.4f}\n")

        else:
            counter += 1
            print(f"❌ No Improvement: {counter}/{patience}\n")

        scheduler.step()
        
        
        # ====================================
        # SAVE CHECKPOINT
        # ====================================
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_val_acc': best_val_acc,
            'counter': counter
        }
        torch.save(checkpoint, checkpoint_path)
        print("💾 Checkpoint Saved!")

        print(f"🔁 Completed Epoch {epoch+1}/{num_epochs}\n")

        if counter >= patience:
            print("⛔ Early Stopping Triggered!")
            break

    # ====================================
    # SAVE LAST MODEL
    # ====================================
    torch.save(model.state_dict(), os.path.join(model_save_path, "last_model.pth"))
    print("\n✅ Training Finished!")



    # ====================================
    # 🎯 FINAL EVALUATION & EXPORT (ADDED HERE)
    # ====================================
    print("\n🔍 Loading best model weights for final evaluation...")
    best_model_path = os.path.join(model_save_path, "best_model.pth")
    
    if os.path.exists(best_model_path):
        # Load the best weights back into your model structure
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Exporting Final Accuracy"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        final_accuracy = correct / max(total, 1)
        
        # Format and save the results
        output_text = (
            f"========================================\n"
            f"🎯 FINAL RUN RESULTS\n"
            f"========================================\n"
            f"Verified Validation Accuracy: {final_accuracy * 100:.2f}%\n"
            f"Raw Score: {correct}/{total} correct images.\n"
            f"========================================\n"
        )
        
        print(f"\n📊 Results:\n{output_text}")
        
        # Write to local file
        with open("evaluation_accuracy.txt", "w") as f:
            f.write(output_text)
        print("💾 Exported accuracy verification to evaluation_accuracy.txt")
    else:
        print("❌ Could not find best_model.pth to evaluate.")


if __name__ == '__main__':
    main()