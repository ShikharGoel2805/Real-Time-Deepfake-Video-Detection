import os, json, yaml, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from src.model import build_model

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class ImgDS(Dataset):
    def __init__(self, files, tfm):
        self.files = files; self.tfm = tfm
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        p, y = self.files[i]
        img = Image.open(p).convert("RGB")
        x = self.tfm(image=np.array(img))["image"]
        return x, torch.tensor(y, dtype=torch.float32)

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
        return A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()])

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    set_seed(cfg["seed"])

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["model"]["backbone"], cfg["model"]["num_classes"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = GradScaler(enabled=cfg["train"]["mixed_precision"])

    best_auc = 0.0
    ckpt_dir = Path(cfg["train"]["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with autocast(enabled=cfg["train"]["mixed_precision"]):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * xb.size(0)
        scheduler.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs.tolist()); gts.extend(yb.numpy().tolist())
        try:
            auc = roc_auc_score(gts, preds)
        except:
            auc = float("nan")
        print(f"Epoch {epoch:02d} | train_loss={(running/len(train_ds)):.4f} | val_auc={auc:.4f}")

        if auc == auc and auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "auc": auc}, ckpt_dir/"best.pt")
            print(f"  ✓ Saved new best (AUC={auc:.4f})")
