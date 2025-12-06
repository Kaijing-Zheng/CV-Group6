# evaluate_asl.py
import argparse
import json
from pathlib import Path
import csv

import numpy as np
import cv2
from PIL import Image
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# --- MediaPipe hand detector ---
import mediapipe as mp
mp_hands = mp.solutions.hands


# ------------------ image + feature helpers (same math as training) ------------------
def read_image_bgr(path: Path, max_side: int | None = 512) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = img.size
        scale = max(w, h) / max_side
        if scale > 1.0:
            img = img.resize((int(round(w / scale)), int(round(h / scale))), Image.BILINEAR)
    arr = np.array(img)  # RGB
    return arr[:, :, ::-1].copy()  # BGR


def extract_hand_landmarks(img_bgr: np.ndarray, hands_detector: mp_hands.Hands) -> np.ndarray | None:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)
    if not result.multi_hand_landmarks or not result.multi_handedness:
        return None
    # best hand by score
    scores = [h.classification[0].score for h in result.multi_handedness]
    idx = int(np.argmax(scores))
    hand_lms = result.multi_hand_landmarks[idx]
    pts = [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
    return np.asarray(pts, dtype=np.float32)  # (21,3)


def clean_landmarks(pts_xyz: np.ndarray) -> np.ndarray:
    pts_xy = pts_xyz[:, :2]
    center = pts_xy.mean(axis=0, keepdims=True)
    pts_centered = pts_xy - center
    flat = pts_centered.reshape(-1)
    denom = np.max(np.abs(flat))
    if denom < 1e-12:
        denom = 1.0
    return (flat / denom).astype(np.float32)  # (42,)


# ------------------ simple enhancement tries to help MP find the hand ------------------
def enhancement_variants(img_bgr: np.ndarray) -> list[np.ndarray]:
    h, w = img_bgr.shape[:2]

    # 1) original
    variants = [img_bgr]

    # 2) horizontal flip
    variants.append(cv2.flip(img_bgr, 1))

    # 3) CLAHE on L channel (LAB)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    variants.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

    # 4) Gamma brighten
    gamma = 0.7  # <1 brightens
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    variants.append(cv2.LUT(img_bgr, table))

    # 5) Slight contrast/brightness
    variants.append(cv2.convertScaleAbs(img_bgr, alpha=1.25, beta=5))

    # 6) Median denoise
    variants.append(cv2.medianBlur(img_bgr, 3))

    return variants


# ------------------ evaluation helpers ------------------
def infer_label_from_filename(fname: str) -> str | None:
    # Works with "A_test.jpg", "space_test.jpg", etc.
    stem = Path(fname).stem  # e.g., "A_test"
    part = stem.split("_")[0].lower()  # "a" or "space"
    # normalize to label names list format (case sensitive there)
    special = {"space": "space", "nothing": "nothing", "del": "del"}
    if part in special:
        return special[part]
    if len(part) == 1 and part.isalpha():
        return part.upper()
    return None


def plot_confusion_matrix(cm, class_names, out_path):
    fig = plt.figure(figsize=(9, 9))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)

    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)

    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_folder(
    images_dir: Path,
    labels_path: Path,
    model_path: Path,
    pattern: str = "*.jpg",
    max_side: int | None = 512,
    min_detection_confidence: float = 0.5,
    static_image_mode: bool = True,
    csv_out: Path | None = None,
    cm_out: Path | None = None,
):
    # Load labels list
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))["labels"]
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    # Load trained classifier pipeline
    clf = joblib.load(model_path)

    # Collect images
    imgs = sorted(images_dir.glob(pattern))
    if not imgs:
        raise SystemExit(f"No images matching '{pattern}' under {images_dir}")

    y_true, y_pred, filenames = [], [], []
    detected_flags = []  # True if a hand was detected (any variant)

    with mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
    ) as hands:
        for p in imgs:
            true_lab = infer_label_from_filename(p.name)
            if true_lab is None or true_lab not in label_to_id:
                # Skip images we cannot assign a ground truth from filename
                continue
            true_id = label_to_id[true_lab]

            img_bgr0 = read_image_bgr(p, max_side=max_side)
            pred_id = None
            detected = False

            for var in enhancement_variants(img_bgr0):
                pts = extract_hand_landmarks(var, hands)
                if pts is None:
                    continue
                detected = True
                feat = clean_landmarks(pts).reshape(1, -1)  # (1,42)
                pred_id = int(clf.predict(feat)[0])
                break  # stop at first successful detection

            if pred_id is None:
                # mark as "NoHand" pseudo-class for analysis; treat as incorrect
                pred_id = -1

            y_true.append(true_id)
            y_pred.append(pred_id)
            filenames.append(p.name)
            detected_flags.append(detected)

    # Compute metrics
    n = len(y_true)
    detected_mask = np.array(detected_flags, dtype=bool)
    valid_pred_mask = (np.array(y_pred) >= 0)

    overall_correct = sum((np.array(y_pred) == np.array(y_true)) & valid_pred_mask)
    overall_acc = overall_correct / n if n else 0.0

    if detected_mask.any():
        det_correct = sum((np.array(y_pred) == np.array(y_true)) & detected_mask)
        detected_acc = det_correct / detected_mask.sum()
    else:
        detected_acc = 0.0

    # Confusion matrix only over successful detections
    y_true_det = np.array(y_true)[valid_pred_mask]
    y_pred_det = np.array(y_pred)[valid_pred_mask]
    cm = confusion_matrix(y_true_det, y_pred_det, labels=list(range(len(labels))))

    # Classification report (detected only)
    cls_rep = classification_report(
        y_true_det, y_pred_det, labels=list(range(len(labels))), target_names=labels, digits=4
    )

    # Save outputs
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_label", "pred_label", "detected"])
            for fn, t, p, d in zip(filenames, y_true, y_pred, detected_flags):
                t_lab = labels[t]
                p_lab = labels[p] if p >= 0 else "NoHand"
                w.writerow([fn, t_lab, p_lab, int(d)])

    if cm_out is not None:
        plot_confusion_matrix(cm, labels, cm_out)

    # Print summary
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Images evaluated: {n}")
    print(f"  • Hand detected in: {detected_mask.sum()} images")
    print(f"  • No-hand (all variants failed): {(~detected_mask).sum()} images")
    print(f"\nOverall accuracy (counting 'NoHand' as wrong): {overall_acc:.4f}")
    print(f"Accuracy on detected images only:              {detected_acc:.4f}")
    print("\nClassification report (detected images only):")
    print(cls_rep)
    print("====================================================\n")


def main():
    ap = argparse.ArgumentParser(description="Evaluate ASL classifier on a folder of test images")
    ap.add_argument("--images_dir", type=str, required=True, help="Folder with test images (e.g. asl_alphabet_test)")
    ap.add_argument("--labels", type=str, required=True, help="Path to labels.json from pipeline")
    ap.add_argument("--model", type=str, required=True, help="Path to trained model .joblib")
    ap.add_argument("--pattern", type=str, default="*.jpg", help="Filename pattern (default: *.jpg)")
    ap.add_argument("--max_side", type=int, default=512, help="Resize long side to this before detection")
    ap.add_argument("--min_conf", type=float, default=0.5, help="MediaPipe min_detection_confidence")
    ap.add_argument("--static_image_mode", action="store_true", help="Use MP static image mode")
    ap.add_argument("--csv_out", type=str, default="eval_results.csv", help="Per-image results CSV")
    ap.add_argument("--cm_out", type=str, default="confusion_matrix.png", help="Confusion matrix image")
    args = ap.parse_args()

    evaluate_folder(
        images_dir=Path(args.images_dir),
        labels_path=Path(args.labels),
        model_path=Path(args.model),
        pattern=args.pattern,
        max_side=args.max_side,
        min_detection_confidence=args.min_conf,
        static_image_mode=args.static_image_mode,
        csv_out=Path(args.csv_out) if args.csv_out else None,
        cm_out=Path(args.cm_out) if args.cm_out else None,
    )


if __name__ == "__main__":
    main()
