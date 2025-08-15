import json
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

def gather_files(frames_root):
    files = []
    for label, y in [("real", 0), ("fake", 1)]:
        for p in Path(frames_root, label).rglob("*.jpg"):
            files.append((str(p), y))
    return files

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    frames_root = cfg["data"]["frames_root"]
    files = gather_files(frames_root)
    if not files:
        raise SystemExit(f"No crops found under {frames_root}. Run preprocessing first.")
    y = [lbl for _, lbl in files]
    train, val = train_test_split(
        files, test_size=1-cfg["data"]["train_split"], stratify=y, random_state=cfg["seed"]
    )
    Path("cache").mkdir(exist_ok=True)
    json.dump(train, open("cache/train.json","w"))
    json.dump(val,   open("cache/val.json","w"))
    print(f"Saved: {len(train)} train, {len(val)} val")
