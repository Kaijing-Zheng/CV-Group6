
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

# Mediapipe is required. Install with: pip install mediapipe==0.10.14 opencv-python pillow tqdm
import mediapipe as mp
import cv2
mp_hands = mp.solutions.hands

def list_images_by_label(root: Path, patterns: List[str]) -> Dict[str, List[Path]]:
    """
    Assumes directory structure like:
      root/
        A/*.jpg
        B/*.png
        ...
    Returns a dict: label -> list of image paths
    """
    label_to_files = {}
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = []
        for pat in patterns:
            files.extend(sorted(label_dir.glob(pat)))
        if files:
            label_to_files[label_dir.name] = files
    return label_to_files

def extract_hand_landmarks(img_bgr: np.ndarray,
                           hands_detector: "mp.solutions.hands.Hands".Hands) -> Optional[np.ndarray]:
    """
    Returns (21, 3) array of (x,y,z) in normalized image coordinates for the BEST hand
    (highest detection score) or None if no hands.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)
    if not result.multi_hand_landmarks or not result.multi_handedness:
        return None

    # choose the hand with highest score
    scores = [h.classification[0].score for h in result.multi_handedness]
    idx = int(np.argmax(scores))
    hand_lms = result.multi_hand_landmarks[idx]

    pts = []
    for lm in hand_lms.landmark:
        pts.append([lm.x, lm.y, lm.z])
    return np.asarray(pts, dtype=np.float32)  # (21, 3)

def clean_landmarks(pts_xyz: np.ndarray) -> np.ndarray:
    """
    Cleaning as specified:
      - Remove z (depth)
      - Centralize coordinates to the center point of the hand (mean of x,y over 21 landmarks)
      - Flatten to 1D
      - Normalize w.r.t. max absolute value (so values in [-1,1])
    Input: (21,3) float32
    Output: (42,) float32
    """
    pts_xy = pts_xyz[:, :2]  # (21, 2)
    center = pts_xy.mean(axis=0, keepdims=True)  # (1,2)
    pts_centered = pts_xy - center  # (21, 2)
    flat = pts_centered.reshape(-1)  # (42,)
    denom = np.max(np.abs(flat))
    if denom < 1e-12:
        denom = 1.0
    flat_norm = flat / denom
    return flat_norm.astype(np.float32)

def read_image_bgr(path: Path, max_side: Optional[int]=None) -> np.ndarray:
    """
    Read with Pillow to be lenient, convert to BGR for Mediapipe/OpenCV.
    Optionally resize so the longest side == max_side to speed up processing (keeps aspect).
    """
    img = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = img.size
        scale = max(w, h) / max_side
        if scale > 1.0:
            new_w = int(round(w / scale))
            new_h = int(round(h / scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(img)  # RGB
    bgr = arr[:, :, ::-1].copy()
    return bgr

def stratified_split(paths: List[Path], labels: List[int], 
                     train_ratio=0.8, val_ratio=0.2,
                     seed=42):
    """
    Simple stratified split without sklearn.
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    paths = np.array(paths, dtype=object)
    labels = np.array(labels, dtype=int)
    unique = np.unique(labels)
    train_idx, val_idx = [], []
    for c in unique:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
       
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train+n_val])
    return (paths[train_idx].tolist(), labels[train_idx].tolist(), 
            paths[val_idx].tolist(), labels[val_idx].tolist())

def build_dataset(data_root: Path,
                  output_dir: Path,
                  patterns: List[str],
                  max_side: Optional[int],
                  min_confidence: float,
                  static_image_mode: bool,
                  train_ratio: float,
                  val_ratio: float,
                  seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map labels to integer ids
    label_to_files = list_images_by_label(data_root, patterns)
    if not label_to_files:
        raise SystemExit(f"No images found under {data_root}. Expected folder-per-label with images.")

    labels_sorted = sorted(label_to_files.keys())
    label_to_id = {lab: i for i, lab in enumerate(labels_sorted)}
    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels_sorted}, f, indent=2)

    # Flatten file list and labels
    all_paths, all_labels = [], []
    for lab, files in label_to_files.items():
        all_paths.extend(files)
        all_labels.extend([label_to_id[lab]] * len(files))

    # Stratified split by label
    (train_paths, train_labels,
     val_paths, val_labels) = stratified_split(
        all_paths, all_labels, 
        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    def process_split(paths: List[Path], labels: List[int], split_name: str):
        xs, ys, kept = [], [], 0
        # Configure Mediapipe Hands
        with mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_confidence
        ) as hands:
            for p, y in tqdm(zip(paths, labels), total=len(paths), desc=f"Extracting {split_name}"):
                try:
                    img_bgr = read_image_bgr(Path(p), max_side=max_side)
                    pts = extract_hand_landmarks(img_bgr, hands)
                    if pts is None:
                        continue
                    feat = clean_landmarks(pts)  # (42,)
                    xs.append(feat)
                    ys.append(y)
                    kept += 1
                except Exception as e:
                    # Skip unreadable/bad images
                    continue
        if kept == 0:
            raise SystemExit(f"No hands found for split {split_name}. Check data or confidence settings.")
        X = np.stack(xs, axis=0).astype(np.float32)  # (N, 42)
        y = np.asarray(ys, dtype=np.int64)         # (N,)
        np.savez_compressed(output_dir / f"{split_name}.npz", X=X, y=y)
        return kept

    ntr = process_split(train_paths, train_labels, "train")
    nval = process_split(val_paths, val_labels, "val")
    
    # Write a small summary
    summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "labels": labels_sorted,
        "counts_after_extraction": {
            "train": int(ntr),
            "val": int(nval),
        },
        "feature_dim": 42,
        "normalization": "center by mean(x,y), divide by max-abs; z removed"
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

def main():
    ap = argparse.ArgumentParser(description="ASL Approach 2: Landmark extraction & dataset prep")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Root folder containing subfolders per label (e.g., data/raw/A, data/raw/B, ...)")
    ap.add_argument("--output_dir", type=str, default="data/processed",
                    help="Where to write train/val/test .npz, labels.json, summary.json")
    ap.add_argument("--patterns", type=str, nargs="+", default=["*.jpg", "*.jpeg", "*.png", "*.bmp"],
                    help="Glob patterns for image files per label folder")
    ap.add_argument("--max_side", type=int, default=512,
                    help="If set, downscale images so the longest side equals this (speeds up processing)")
    ap.add_argument("--min_confidence", type=float, default=0.5,
                    help="Mediapipe min_detection_confidence")
    ap.add_argument("--static_image_mode", action="store_true",
                    help="Use Mediapipe static image mode (recommended for photos)")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    build_dataset(data_root=data_root,
                  output_dir=output_dir,
                  patterns=args.patterns,
                  max_side=args.max_side,
                  min_confidence=args.min_confidence,
                  static_image_mode=args.static_image_mode,
                  train_ratio=args.train_ratio,
                  val_ratio=args.val_ratio,
                  seed=args.seed)

if __name__ == "__main__":
    main()
