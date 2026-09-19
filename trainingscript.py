# ==============================================================================
# 🚀 AGROSCAN: ALL-IN-ONE PIPELINE (FIXED IMPORT ORDER)
# ==============================================================================

# 1. INSTALL DEPENDENCIES FIRST (Before importing anything else)
import subprocess
import sys

print("📦 Installing libraries (this takes 10-20 seconds)...")
# We use subprocess to ensure installation completes before Python moves on
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "kaggle", "seaborn", "matplotlib", "scikit-learn", "-q"])
print("✅ Libraries installed!")

# ------------------------------------------------------------------------------
# 2. NOW IMPORT EVERYTHING
# ------------------------------------------------------------------------------
import os
import shutil
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from ultralytics import YOLO  # Now this will work
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import files
from getpass import getpass

# ------------------------------------------------------------------------------
# 3. SETUP & GPU CHECK
# ------------------------------------------------------------------------------
print("🚀 Initializing Pipeline...")

if not torch.cuda.is_available():
    raise SystemError("❌ STOP! You are on CPU. Go to Runtime > Change runtime type > T4 GPU")
print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")

# --- ENTER YOUR KAGGLE CREDENTIALS HERE ---
if 'KAGGLE_USERNAME' not in os.environ:
    print("\n🔐 CREDENTIALS REQUIRED")
    os.environ['KAGGLE_USERNAME'] = input("Enter Kaggle Username: ").strip()
    os.environ['KAGGLE_KEY'] = getpass("Enter Kaggle API Key: ").strip()
print("✅ Credentials set!")

# ------------------------------------------------------------------------------
# 4. FAST DATA PREPARATION
# ------------------------------------------------------------------------------
DATASET_DIR = Path("/content/plantvillage_data")

if not DATASET_DIR.exists():
    print("⬇️ Downloading Dataset (Fast Connection)...")
    !kaggle datasets download -d abdallahalidev/plantvillage-dataset

    print("📦 Unzipping to Local Storage (RAM optimized)...")
    !unzip -q plantvillage-dataset.zip -d raw_data

    print("⚡ Organizing Data Structure...")
    source_root = Path("raw_data")
    color_dir = next(source_root.rglob("color"), None)
    if not color_dir: color_dir = source_root

    (DATASET_DIR / "train").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "val").mkdir(parents=True, exist_ok=True)

    classes = [d for d in color_dir.iterdir() if d.is_dir()]
    for cls in classes:
        (DATASET_DIR / "train" / cls.name).mkdir(exist_ok=True)
        (DATASET_DIR / "val" / cls.name).mkdir(exist_ok=True)

        imgs = list(cls.glob("*.*"))
        random.shuffle(imgs)
        split = int(len(imgs) * 0.8)

        for img in imgs[:split]: shutil.move(str(img), DATASET_DIR / "train" / cls.name / img.name)
        for img in imgs[split:]: shutil.move(str(img), DATASET_DIR / "val" / cls.name / img.name)

    print(f"✅ Data Ready: {len(classes)} classes prepared.")
    !rm -rf raw_data plantvillage-dataset.zip

# ------------------------------------------------------------------------------
# 5. TURBO TRAINING
# ------------------------------------------------------------------------------
print("🔥 STARTING HIGH-SPEED TRAINING...")
PROJECT_NAME = 'plantvillage_research'
start_time = time.time()

model = YOLO('yolo11n-cls.pt')

results = model.train(
    data='/content/plantvillage_data',
    epochs=15,
    imgsz=224,
    batch=64,
    workers=4,
    cache=True,
    amp=True,
    project='runs/classify',
    name=PROJECT_NAME,
    exist_ok=True,
    device=0
)

print(f"🏁 Training Finished in {(time.time() - start_time)/60:.1f} minutes!")

# ------------------------------------------------------------------------------
# 6. RESEARCH DATA GENERATION
# ------------------------------------------------------------------------------
print("📊 Generating Research Papers Data...")
OUTPUT_DIR = 'Research_Output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A) Training Curves
results_path = f'runs/classify/{PROJECT_NAME}/results.csv'
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()
    df.to_csv(f"{OUTPUT_DIR}/raw_training_metrics.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/accuracy_top1'] * 100, label='Top-1 Accuracy', linewidth=2)
    plt.plot(df['epoch'], df['metrics/accuracy_top5'] * 100, label='Top-5 Accuracy', linewidth=2, linestyle='--')
    plt.title('Model Accuracy over Epochs'); plt.legend(); plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/training_accuracy_curve.png", dpi=300); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/loss'], label='Train Loss', linewidth=2)
    plt.plot(df['epoch'], df['val/loss'], label='Validation Loss', linewidth=2)
    plt.title('Training & Validation Loss'); plt.legend(); plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/training_loss_curve.png", dpi=300); plt.close()

# B) Confusion Matrix
print("🔍 Running Comprehensive Validation...")
best_model_path = f"runs/classify/{PROJECT_NAME}/weights/best.pt"
model = YOLO(best_model_path)

val_dir = DATASET_DIR / 'val'
class_names = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
y_true, y_pred = [], []

chunk_size = 64
for class_name in class_names:
    class_path = val_dir / class_name
    images = list(class_path.glob('*.*'))
    for i in range(0, len(images), chunk_size):
        chunk = images[i:i + chunk_size]
        results = model(chunk, verbose=False)
        for res in results:
            y_true.append(class_name)
            y_pred.append(class_names[res.probs.top1])

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv(f"{OUTPUT_DIR}/classification_report_full.csv")

cm = confusion_matrix(y_true, y_pred, labels=class_names)
plt.figure(figsize=(20, 15))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix'); plt.xticks(rotation=90); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_high_res.png", dpi=300); plt.close()

# ------------------------------------------------------------------------------
# 7. PACKAGING & DOWNLOAD
# ------------------------------------------------------------------------------
print("📦 Packaging files...")
shutil.copy(best_model_path, f"{OUTPUT_DIR}/best_model.pt")
try:
    model.export(format='onnx')
    shutil.copy(f"runs/classify/{PROJECT_NAME}/weights/best.onnx", f"{OUTPUT_DIR}/best_model.onnx")
except: pass

shutil.make_archive('AgroScan_Research_Package', 'zip', OUTPUT_DIR)
print(f"\n🎉 SUCCESS! 'AgroScan_Research_Package.zip' is ready.")
files.download('AgroScan_Research_Package.zip')