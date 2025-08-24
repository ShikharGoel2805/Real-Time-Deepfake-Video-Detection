# src/train.py
import timm
from torch.amp import GradScaler    # NEW (replaces torch.cuda.amp.GradScaler)
from torch.cuda.amp import autocast  # you can keep this; only GradScaler changed

import os
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.cuda.amp import autocast, GradScaler

# ---- Config (simple args for now) ----
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, default="data/split/train")
    ap.add_argument("--val_dir",   type=str, default="data/split/val")
    ap.add_argument("--backbone",  type=str, default="resnet18", help="resnet18, resnet50, densenet121, etc.")
    ap.add_argument("--epochs",    type=int, default=15)
    ap.add_argument("--batch_size",type=int, default=32)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size",  type=int, default=224)
    ap.add_argument("--patience",  type=int, default=5, help="early stopping patience (epochs)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--seed",      type=int, default=42)
    return ap.parse_args()

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- Utilities ----
class LabelRemapDataset(Dataset):
    """
    Wrap an ImageFolder so labels are mapped to {real:0, fake:1} consistently,
    regardless of folder indexing order inside ImageFolder.
    """
    def __init__(self, imagefolder_ds: datasets.ImageFolder, target_map: dict[str,int]):
        self.ds = imagefolder_ds
        self.target_map = target_map  # e.g., {"real":0, "fake":1}
        # build per-class old->new
        self.idx_to_class = {v:k for k,v in self.ds.class_to_idx.items()}
        self.class_to_new = {old_idx: target_map[self.idx_to_class[old_idx]] for old_idx in self.idx_to_class}

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        x, old_y = self.ds[i]
        y = self.class_to_new[old_y]
        return x, torch.tensor(y, dtype=torch.long)

def build_dataloaders(train_dir, val_dir, img_size, batch_size, num_workers):
    # NOTE: keep transforms simple to match evaluate.py (no ImageNet normalization)
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
        transforms.ToTensor(),  # scales to [0,1]
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_if = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_if   = datasets.ImageFolder(val_dir,   transform=val_tfms)

    # enforce 0=real, 1=fake regardless of alphabetical order
    assert set(train_if.classes) >= {"real", "fake"}, \
        f"Expected 'real' and 'fake' folders under {train_dir}, found: {train_if.classes}"
    target_map = {"real": 0, "fake": 1}

    train_ds = LabelRemapDataset(train_if, target_map)
    val_ds   = LabelRemapDataset(val_if,   target_map)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, train_if.classes

def build_model(backbone: str, num_classes: int = 2):
    """
    Build a model by name. Uses torchvision for common ResNet/DenseNet names,
    otherwise falls back to timm.create_model for anything else.
    """
    tv_backbones = {"resnet18", "resnet50", "densenet121"}
    if backbone in tv_backbones:
        if backbone == "resnet18":
            m = models.resnet18(pretrained=True)
            in_f = m.fc.in_features
            m.fc = nn.Linear(in_f, num_classes)
        elif backbone == "resnet50":
            m = models.resnet50(pretrained=True)
            in_f = m.fc.in_features
            m.fc = nn.Linear(in_f, num_classes)
        elif backbone == "densenet121":
            m = models.densenet121(pretrained=True)
            in_f = m.classifier.in_features
            m.classifier = nn.Linear(in_f, num_classes)
        return m

    # Otherwise: use timm (handles tf_efficientnet_b0_ns, convnext_*, vit_*, etc.)
    m = timm.create_model(
        backbone,
        pretrained=True,
        num_classes=num_classes,   # head replaced for our class count
        drop_rate=0.2              # mild regularization; tweak if desired
    )

    # Some timm models expose different head attributes; ensure correct out dim
    if hasattr(m, "classifier") and isinstance(m.classifier, nn.Linear):
        if m.classifier.out_features != num_classes:
            m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        if m.fc.out_features != num_classes:
            m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif hasattr(m, "head") and isinstance(m.head, nn.Linear):
        if m.head.out_features != num_classes:
            m.head = nn.Linear(m.head.in_features, num_classes)

    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # P(class=1=fake)
        all_probs.extend(probs.tolist())
        all_targets.extend(yb.numpy().astype(int).tolist())

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    prec= precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return acc, auc, f1, prec, rec

def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    train_loader, val_loader, classes = build_dataloaders(
        args.train_dir, args.val_dir, args.img_size, args.batch_size, args.num_workers
    )
    num_classes = 2  # we remapped to real(0)/fake(1)

    model = build_model(args.backbone, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0
    epochs_no_improve = 0

    print(f"Starting training on {device} | backbone={args.backbone} | classes={classes}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, total, correct = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        scheduler.step()
        train_acc = correct / max(1, total)
        train_loss = running_loss / max(1, len(train_loader))

        # Validation
        val_acc, val_auc, val_f1, val_prec, val_rec = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} | val_auc={val_auc:.4f} | val_f1={val_f1:.4f} | "
              f"val_p={val_prec:.4f} | val_r={val_rec:.4f}")

        # Checkpoint on best AUC
        improved = (not np.isnan(val_auc)) and (val_auc > best_auc)
        if improved:
            best_auc = val_auc
            name = f"{args.backbone}_best.pth"
            torch.save({"model": model.state_dict(), "epoch": epoch, "metric": {"auc": val_auc}},
                       ckpt_dir / name)
            print(f"  ✓ Saved new best checkpoint: {ckpt_dir / name} (AUC={val_auc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered (no AUC improvement for {args.patience} epochs).")
            break

    # Save final model in project root for compatibility with your current evaluate flow
    torch.save(model.state_dict(), "final_model.pth")
    print("✅ Training finished. Saved final_model.pth and best checkpoint (if improved).")

if __name__ == "__main__":
    main()
