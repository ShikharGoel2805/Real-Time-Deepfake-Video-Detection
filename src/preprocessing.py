import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN  # switched to facenet_pytorch

def ensure_dirs(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def list_media(root, exts):
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for e in exts:
        out.extend(root.rglob(f"*{e}"))
    return sorted(out)

def detect_and_save_faces(img_bgr, save_dir, stem, detector, max_faces=4):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, _ = detector.detect(img_rgb)
    count = 0
    if boxes is not None:
        for i, (x1, y1, x2, y2) in enumerate(boxes[:max_faces]):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out_path = save_dir / f"{stem}_{i:02d}.jpg"
            cv2.imwrite(str(out_path), crop)
            count += 1
    return count

def process_videos(in_root, out_root, every_n=5, max_frames=None, detector=None):
    total_saved = 0
    for label in ["real", "fake"]:
        vids = list_media(Path(in_root)/label, [".mp4", ".avi", ".mov", ".mkv"])
        for vp in tqdm(vids, desc=f"[Video] {label}"):
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                continue
            frame_idx, saved_here = 0, 0
            save_dir = Path(out_root)/label/vp.stem
            ensure_dirs(save_dir)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % every_n == 0:
                    if max_frames and saved_here >= max_frames:
                        break
                    saved = detect_and_save_faces(
                        frame, save_dir, f"{vp.stem}_{frame_idx:06d}", detector
                    )
                    saved_here += saved
                    total_saved += saved
                frame_idx += 1
            cap.release()
    return total_saved

def process_images(in_root, out_root, detector=None):
    total_saved = 0
    for label in ["real", "fake"]:
        imgs = list_media(Path(in_root)/label, [".jpg", ".jpeg", ".png"])
        for ip in tqdm(imgs, desc=f"[Image] {label}"):
            img = cv2.imread(str(ip))
            if img is None:
                continue
            save_dir = Path(out_root)/label/ip.stem
            ensure_dirs(save_dir)
            total_saved += detect_and_save_faces(img, save_dir, ip.stem, detector)
    return total_saved

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))

    raw_videos = cfg["data"]["raw_videos"]
    raw_images = cfg["data"]["raw_images"]
    frames_root = cfg["data"]["frames_root"]
    det_size = cfg["preprocess"]["detector_image_size"]
    every_n = cfg["preprocess"]["sample_every_n"]
    max_frames = cfg["preprocess"]["max_frames_per_video"]

    ensure_dirs(frames_root)
    detector = MTCNN(image_size=det_size, margin=20, keep_all=True)

    # Process videos if "real" or "fake" subfolders exist
    have_videos = (Path(raw_videos)/"real").exists() or (Path(raw_videos)/"fake").exists()
    total_v = process_videos(raw_videos, frames_root, every_n, max_frames, detector) if have_videos else 0

    # Process images if "real" or "fake" subfolders exist
    have_images = (Path(raw_images)/"real").exists() or (Path(raw_images)/"fake").exists()
    total_i = process_images(raw_images, frames_root, detector) if have_images else 0

    print(f"Done. Saved crops: from videos={total_v}, from images={total_i}.")
