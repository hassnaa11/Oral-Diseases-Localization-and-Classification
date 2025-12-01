
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import json

# ==================== CONFIG ====================
DEVICE = 0
MODEL_NAME = "yolo11x.pt"  
DATA_YAML = "/home/moabouag/far/oral-detect/oral_yolo_dataset/Data_balanced/data.yaml"

EPOCHS = 400
IMGSZ = 1280
BATCH = 8
PATIENCE = 80

RUN_NAME = f"YOLOv11x_ORAL_SOTA_{datetime.now().strftime('%Y%m%d_%H%M')}"
PROJECT_DIR = "oral_runs"
OUTPUT_DIR = Path(PROJECT_DIR) / RUN_NAME
PLOTS_DIR = OUTPUT_DIR / "progress_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== HISTORY STORAGE ====================
history = {
    "epoch": [], "mAP50": [], "mAP50_95": [], "precision": [], "recall": [],
    "box_loss": [], "cls_loss": [], "dfl_loss": []
}

# ==================== CALLBACK ====================
def on_train_epoch_end(trainer):
    epoch = trainer.epoch + 1
    
# Extract validation metrics (available after val runs)
    metrics = trainer.metrics
    mAP50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    mAP = float(metrics.get("metrics/mAP50-95(B)", 0.0))
    precision = float(metrics.get("metrics/precision(B)", 0.0))
    recall = float(metrics.get("metrics/recall(B)", 0.0))

    # Loss from last training batch

    losses = trainer.loss_items
    if losses is not None and len(losses) >= 3:
        box_l = float(losses[0].item())
        cls_l = float(losses[1].item())
        dfl_l = float(losses[2].item())
    else:
        box_l = cls_l = dfl_l = 0.0

    # Save to history
    history["epoch"].append(epoch)
    history["mAP50"].append(mAP50)
    history["mAP50_95"].append(mAP)
    history["precision"].append(precision)
    history["recall"].append(recall)
    history["box_loss"].append(box_l)
    history["cls_loss"].append(cls_l)
    history["dfl_loss"].append(dfl_l)

    # Live print
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch}/{EPOCHS} - VALIDATION RESULTS")
    print(f"mAP@0.5       : {mAP50:.4f}  {'Excellent' if mAP50 > 0.70 else 'Great' if mAP50 > 0.60 else 'Needs Improvement'}")
    print(f"mAP@0.5:0.95  : {mAP:.4f}")
    print(f"Precision     : {precision:.4f} | Recall : {recall:.4f}")
    print(f"Losses → Box:{box_l:.4f}  Cls:{cls_l:.4f}  DFL:{dfl_l:.4f}")
    print(f"{'='*80}\n")

    # Save live plot every validation
    if len(history["epoch"]) > 1:
        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.plot(history["epoch"], history["mAP50"], 'g.-', linewidth=2, label="mAP@0.5")
        plt.plot(history["epoch"], history["mAP50_95"], 'm.-', linewidth=2, label="mAP@0.5:0.95")
        plt.title("Validation mAP Progress", fontsize=16, pad=20)
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)

        plt.subplot(1, 2, 2)
        plt.plot(history["epoch"], history["box_loss"], label="Box Loss", alpha=0.8)
        plt.plot(history["epoch"], history["cls_loss"], label="Cls Loss", alpha=0.8)
        plt.plot(history["epoch"], history["dfl_loss"], label="DFL Loss", alpha=0.8)
        plt.title("Training Losses", fontsize=16, pad=20)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "live_progress.png", dpi=300, bbox_inches='tight')
        plt.savefig(PLOTS_DIR / f"epoch_{epoch:03d}.png", dpi=300, bbox_inches='tight')
        plt.close()

# ==================== LOAD MODEL & TRAIN ====================
model = YOLO(MODEL_NAME)
model.add_callback("on_train_epoch_end", on_train_epoch_end)

print(f"Starting SOTA Oral Disease Training")
print(f"Run: {RUN_NAME}")
print(f"Dataset: Balanced (2953 train + 299 val)")
print(f"Model: {MODEL_NAME} | ImgSize: {IMGSZ} | Batch: {BATCH}\n")

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.008,
    weight_decay=0.0005,
    warmup_epochs=5,
    freeze=10,

    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=15, translate=0.15, scale=0.9, shear=3.0,
    flipud=0.5, fliplr=0.5,
    mosaic=1.0, mixup=0.4, copy_paste=0.4,
    close_mosaic=15,

    box=8.0, cls=0.5, dfl=1.5,
    cache="ram",
    device=DEVICE,
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    pretrained=True,
    verbose=True,
    plots=True,
    save_period=20,
)

# ==================== TTA EVALUATION ====================
best_pt = OUTPUT_DIR / "weights" / "best.pt"
print(f"\nLoading best model: {best_pt}")
model = YOLO(best_pt)

print("Running final evaluation with Test-Time Augmentation (TTA)...")
final = model.val(
    data=DATA_YAML,
    imgsz=IMGSZ,
    batch=1,
    augment=True,           # TTA ON
    conf=0.001,
    iou=0.6,
    device=DEVICE,
    save_json=True,
    plots=True
)

# ==================== SAVE FINAL REPORT ====================
report = {
    "run_name": RUN_NAME,
    "final_mAP50": round(float(final.box.map50), 4),
    "final_mAP50_95": round(float(final.box.map), 4),
    "precision": round(float(final.box.mp), 4),
    "recall": round(float(final.box.mr), 4),
    "per_class_AP50": {final.names[i]: round(float(final.box.ap50[i]), 4) for i in range(len(final.names))},
    "best_model_path": str(best_pt)
}

with open(OUTPUT_DIR / "FINAL_REPORT.json", "w") as f:
    json.dump(report, f, indent=4)

# Final big plot
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(history["epoch"], history["mAP50"], 'g.-', linewidth=3, markersize=8, label="mAP@0.5")
plt.plot(history["epoch"], history["mAP50_95"], 'purple', linewidth=3, label="mAP@0.5:0.95")
plt.title("Final Validation Performance", fontsize=18, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("mAP", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.plot(history["epoch"], history["box_loss"], label="Box Loss", linewidth=2)
plt.plot(history["epoch"], history["cls_loss"], label="Classification Loss", linewidth=2)
plt.plot(history["epoch"], history["dfl_loss"], label="DFL Loss", linewidth=2)
plt.title("Training Loss Curves", fontsize=18, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

plt.suptitle("YOLOv11x - Oral Disease Detection - SOTA Training", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "FINAL_TRAINING_SUMMARY.png", dpi=400, bbox_inches='tight')
plt.close()

# ==================== FINAL MESSAGE ====================
print("\n" + "█" * 90)
print(f"TRAINING COMPLETED SUCCESSFULLY!")
print(f"Final mAP@0.5      : {final.box.map50:.4f}")
print(f"Final mAP@0.5:0.95 : {final.box.map:.4f}")
print(f"Best model saved   : {best_pt}")
print(f"Full report        : {OUTPUT_DIR}/FINAL_REPORT.json")
print(f"Progress plots     : {PLOTS_DIR}")
print(f"Summary graph      : {OUTPUT_DIR}/FINAL_TRAINING_SUMMARY.png")
print("█" * 90)