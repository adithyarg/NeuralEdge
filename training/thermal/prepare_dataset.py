"""
prepare_dataset.py — Extract 32x32 patches from LLVIP infrared images.

Expected LLVIP folder structure (download from https://bupt-ai-cz.github.io/LLVIP/):
    LLVIP/
    ├── infrared/
    │   ├── train/   *.jpg
    │   └── test/    *.jpg
    └── Annotations/ *.xml   (Pascal VOC format, one xml per image)

Output:
    training/thermal/data/patches_train.npz  — X: (N,1024) uint8, y: (N,) int
    training/thermal/data/patches_test.npz

Labels:
    1 = human present (positive patch centred on a person bbox)
    0 = background    (negative patch with no person overlap)

Usage:
    python training/thermal/prepare_dataset.py --llvip_dir /path/to/LLVIP
"""

import argparse
import os
import xml.etree.ElementTree as ET
import random
import numpy as np
import cv2

PATCH_SIZE  = 32        # 32x32 → 1024 pixels, matches our accelerator
NEG_PER_IMG = 3         # background patches per image
IOU_THRESH  = 0.15      # reject negatives with IoU > this against any bbox
RANDOM_SEED = 42


def parse_annotations(xml_path):
    """Return list of (xmin, ymin, xmax, ymax) for all 'person' objects."""
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            if name == "person":
                bb = obj.find("bndbox")
                x1 = int(float(bb.find("xmin").text))
                y1 = int(float(bb.find("ymin").text))
                x2 = int(float(bb.find("xmax").text))
                y2 = int(float(bb.find("ymax").text))
                boxes.append((x1, y1, x2, y2))
    except Exception:
        pass
    return boxes


def iou(boxA, boxB):
    """Intersection over union of two (x1,y1,x2,y2) boxes."""
    ix1 = max(boxA[0], boxB[0])
    iy1 = max(boxA[1], boxB[1])
    ix2 = min(boxA[2], boxB[2])
    iy2 = min(boxA[3], boxB[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(aA + aB - inter)


def extract_patch(img, cx, cy, size):
    """Extract a square patch centred at (cx, cy), return None if out of bounds."""
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + size, y1 + size
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None
    patch = img[y1:y2, x1:x2]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def process_split(img_dir, ann_dir, neg_per_img, rng):
    X, y = [], []
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        xml_path = os.path.join(ann_dir, fname.replace(".jpg", ".xml"))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        boxes = parse_annotations(xml_path) if os.path.exists(xml_path) else []

        # ── Positive patches ─────────────────────────────────────────────────
        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            patch = extract_patch(img, cx, cy, PATCH_SIZE)
            if patch is not None:
                X.append(patch.flatten())
                y.append(1)

        # ── Negative patches ─────────────────────────────────────────────────
        h, w = img.shape
        half  = PATCH_SIZE // 2
        added = 0
        attempts = 0
        while added < neg_per_img and attempts < 200:
            attempts += 1
            cx = rng.randint(half, w - half)
            cy = rng.randint(half, h - half)
            patch_box = (cx-half, cy-half, cx+half, cy+half)
            # reject if overlaps any person bbox
            if any(iou(patch_box, b) > IOU_THRESH for b in boxes):
                continue
            patch = extract_patch(img, cx, cy, PATCH_SIZE)
            if patch is not None:
                X.append(patch.flatten())
                y.append(0)
                added += 1

    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvip_dir", required=True,
                    help="Path to LLVIP root folder")
    ap.add_argument("--out_dir", default="training/thermal/data",
                    help="Output directory for .npz files")
    ap.add_argument("--neg_per_img", type=int, default=NEG_PER_IMG)
    args = ap.parse_args()

    rng = random.Random(RANDOM_SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    ann_dir = os.path.join(args.llvip_dir, "Annotations")

    for split in ("train", "test"):
        img_dir = os.path.join(args.llvip_dir, "infrared", split)
        if not os.path.isdir(img_dir):
            print(f"[WARN] {img_dir} not found, skipping {split}")
            continue

        print(f"Processing {split} ...")
        X, y = process_split(img_dir, ann_dir, args.neg_per_img, rng)

        pos = int(y.sum())
        neg = len(y) - pos
        print(f"  {split}: {len(y)} patches  ({pos} positive, {neg} negative)")

        out_path = os.path.join(args.out_dir, f"patches_{split}.npz")
        np.savez_compressed(out_path, X=X, y=y)
        print(f"  Saved → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
