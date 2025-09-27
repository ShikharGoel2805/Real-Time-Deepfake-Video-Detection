# src/train.py
import timm
import os
import argparse
import yaml   ### NEW
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.cuda.amp import autocast, GradScaler

from model import build_model   ### NEW (import your model.py)

# ---- Config ----
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="default.yaml", help="Path to YAML config")  ### NEW
    ap.add_argument("--backbone", type=str, default=None, help="Override model backbone")
    ap.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--checkpoint_dir", type=str, default=None)
    return ap.parse_args()

def load_config(yaml_path, cli_args):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Flatten useful parts
    merged = {
        "train_dir": "data/split/train",
        "val_dir": "data/split/val",
        "backbone": cfg["model"]["backbone"],
        "epochs": cfg["train"]["epochs"],
        "batch_size": cfg["data"]["batch_size"],
        "lr": cfg["train"]["lr"],
        "weight_decay": cfg["train"]["weight_decay"],
        "img_size": cfg["data"]["img_size"],
        "patience": cfg["train"].get("patience", 5),
        "num_workers": cfg["data"]["num_workers"],
        "checkpoint_dir": cfg["train"]["checkpoint_dir"],
        "seed": cfg.get("seed", 42),
    }

    # Override with CLI if provided
    if cli_args.backbone: merged["backbone"] = cli_args.backbone
    if cli_args.epochs: merged["epochs"] = cli_args.epochs
    if cli_args.batch_size: merged["batch_size"] = cli_args.batch_size
    if cli_args.lr: merged["lr"] = cli_args.lr
    if cli_args.weight_decay: merged["weight_decay"] = cli_args.weight_decay
    if cli_args.img_size: merged["img_size"] = cli_args.img_size
    if cli_args.checkpoint_dir: merged["checkpoint_dir"] = cli_args.checkpoint_dir

    return merged

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- Utilities ----
class LabelRemapDataset(Dataset):
    def __init__(self, imagefolder_ds: datasets.ImageFolder, target_map):
        self.ds = imagefolder_ds
        self.idx_to_class = {v: k for k, v in self.ds.class_to_idx.items()}
        self.class_to_new = {old_idx: target_map[self.idx_to_class[old_idx]] for old_idx in self.idx_to_class}

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        x, old_y = self.ds[i]
        y = self.class_to_new[old_y]
        return x, torch.tensor(y, dtype=torch.long)

def build_dataloaders(train_dir, val_dir, img_size, batch_size, num_workers):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
        transforms.ToTensor(),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_if = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_if = datasets.ImageFolder(val_dir, transform=val_tfms)

    assert set(train_if.classes) >= {"real", "fake"}, f"Expected 'real' and 'fake' folders under {train_dir}"
    target_map = {"real": 0, "fake": 1}

    train_ds = LabelRemapDataset(train_if, target_map)
    val_ds = LabelRemapDataset(val_if, target_map)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, train_if.classes

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(yb.cpu().numpy().astype(int).tolist())

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return acc, auc, f1, prec, rec

def main():
    cli_args = get_args()
    cfg = load_config(cli_args.config, cli_args)   ### NEW
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler() if torch.cuda.is_available() else None

    train_loader, val_loader, classes = build_dataloaders(
        cfg["train_dir"], cfg["val_dir"], cfg["img_size"], cfg["batch_size"], cfg["num_workers"]
    )

    model = build_model(cfg["backbone"], num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg["epochs"]))

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc, epochs_no_improve = 0.0, 0

    print(f"Starting training on {device} | backbone={cfg['backbone']} | classes={classes}")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss, total, correct = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        scheduler.step()
        train_acc = correct / max(1, total)
        train_loss = running_loss / max(1, len(train_loader))

        val_acc, val_auc, val_f1, val_prec, val_rec = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} | val_auc={val_auc:.4f} | val_f1={val_f1:.4f} | "
              f"val_p={val_prec:.4f} | val_r={val_rec:.4f}")

        if (not np.isnan(val_auc)) and (val_auc > best_auc):
            best_auc = val_auc
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "metric": {"auc": val_auc}},
                       ckpt_dir / f"{cfg['backbone']}_best.pth")
            print(f"  ✓ Saved new best checkpoint (AUC={val_auc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg["patience"]:
            print(f"Early stopping (no AUC improvement for {cfg['patience']} epochs).")
            break

    torch.save(model.state_dict(), "final_model.pth")
    print("✅ Training finished. Saved final_model.pth and best checkpoint (if improved).")

if __name__ == "__main__":
    main()
