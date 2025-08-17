import os, json, yaml, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.cuda.amp import autocast, GradScaler
from src.model import build_model

# --------------------------- Utilities ---------------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------- Dataset ---------------------------
class ImgDS(Dataset):
    def __init__(self, files, tfm):
        self.files = files
        self.tfm = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p, y = self.files[i]
        img = Image.open(p).convert("RGB")
        x = self.tfm(image=np.array(img))["image"]
        return x, torch.tensor(y, dtype=torch.float32)

# --------------------------- Transforms ---------------------------
def build_tfms(size, train=True):
    if train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.OneOf([A.GaussianBlur(3, p=1.0), A.MotionBlur(p=1.0)], p=0.2),
            A.JpegCompression(quality_lower=40, quality_upper=90, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(),
            ToTensorV2()
        ])

# --------------------------- Main Training ---------------------------
if __name__ == "__main__":
    # Load config
    cfg = yaml.safe_load(open("configs/default.yaml"))
    set_seed(cfg["seed"])

    # Load dataset
    train_files = json.load(open("cache/train.json"))
    val_files   = json.load(open("cache/val.json"))

    train_tfms = build_tfms(cfg["data"]["img_size"], True)
    val_tfms   = build_tfms(cfg["data"]["img_size"], False)

    train_ds = ImgDS(train_files, train_tfms)
    val_ds   = ImgDS(val_files,   val_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["data"]["batch_size"], shuffle=False,
                              num_workers=cfg["data"]["num_workers"], pin_memory=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = build_model(cfg["model"]["backbone"], cfg["model"]["num_classes"]).to(device)

    # Loss function
    if cfg["model"]["num_classes"] == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = GradScaler(enabled=cfg["train"]["mixed_precision"])

    # Checkpoint directory
    best_metric = 0.0
    ckpt_dir = Path(cfg["train"]["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, cfg["train"]["epochs"]+1):
        # ---------------- Train ----------------
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if cfg["model"]["num_classes"] == 1:
                yb = yb.unsqueeze(1)

            optimizer.zero_grad()
            with autocast(enabled=cfg["train"]["mixed_precision"]):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()

        # ---------------- Validation ----------------
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                if cfg["model"]["num_classes"] == 1:
                    logits = logits.squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds.extend(probs.tolist())
                    gts.extend(yb.numpy().tolist())
                else:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    pred_labels = np.argmax(probs, axis=1)
                    preds.extend(pred_labels.tolist())
                    gts.extend(yb.numpy().astype(int).tolist())

        # Compute metrics
        if cfg["model"]["num_classes"] == 1:
            try:
                metric = roc_auc_score(gts, preds)
            except:
                metric = float("nan")
        else:
            metric = accuracy_score(gts, preds)

        print(f"Epoch {epoch:02d} | train_loss={(running_loss/len(train_ds)):.4f} | val_metric={metric:.4f}")

        # Save best
        if metric == metric and metric > best_metric:
            best_metric = metric
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "metric": metric
            }, ckpt_dir/"best.pt")
            print(f"  ✓ Saved new best (val_metric={metric:.4f})")
