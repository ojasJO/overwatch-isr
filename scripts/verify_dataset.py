"""
verify_dataset.py  —  Sanity-check the YOLO-format xView dataset
=================================================================
Picks N random tiles from train/, draws their YOLO bounding boxes
on top of the image, and saves them to a 'verify' folder so you can
eyeball that the conversion is correct BEFORE training for hours.

Run AFTER prepare_xview.py.

Usage:
    cd ~/uav_pipeline
    source venv/bin/activate
    python3 scripts/verify_dataset.py
    # then open ~/uav_pipeline/datasets/xview_yolo/verify/ and inspect
"""

import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from xview_classes import OPERATIONAL_CLASSES

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOME           = Path.home()
DATASET_ROOT   = HOME / "uav_pipeline" / "datasets" / "xview_yolo"
N_SAMPLES      = 30                # how many tiles to visualize
OUTPUT_DIR     = DATASET_ROOT / "verify"
SPLIT_TO_CHECK = "train"            # or "val"

# Per-class colours (BGR)
COLOURS = {
    0: (80, 200, 80),     # small_vehicle  → green
    1: (0,  140, 255),    # large_vehicle  → orange
    2: (255, 140, 0),     # aircraft       → blue
    3: (180, 80, 255),    # vessel         → pink
    4: (200, 200, 200),   # building       → grey
}


def draw_yolo_labels(img_path: Path, lbl_path: Path) -> "np.ndarray | None":
    """Read image + YOLO labels, return image with boxes drawn."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    H, W = img.shape[:2]

    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        # Empty tile (negative example)
        cv2.putText(img, "EMPTY TILE (negative example)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        return img

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = (int(parts[0]),) + tuple(float(p) for p in parts[1:])

            # YOLO normalised → pixel
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)

            colour = COLOURS.get(cls_id, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

            label = f"{cls_id}:{OPERATIONAL_CLASSES.get(cls_id, '?')}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)),
                          (x1 + tw + 4, y1), colour, -1)
            cv2.putText(img, label, (x1 + 2, max(th, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def main() -> None:
    img_dir = DATASET_ROOT / "images" / SPLIT_TO_CHECK
    lbl_dir = DATASET_ROOT / "labels" / SPLIT_TO_CHECK

    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found. Run prepare_xview.py first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tiles = list(img_dir.glob("*.jpg"))
    print(f"Found {len(all_tiles)} tiles in {SPLIT_TO_CHECK}/")

    if len(all_tiles) == 0:
        print("ERROR: No tiles found.")
        sys.exit(1)

    sample = random.sample(all_tiles, min(N_SAMPLES, len(all_tiles)))
    print(f"Drawing labels on {len(sample)} random samples …")

    n_ok = 0
    class_counts = {k: 0 for k in OPERATIONAL_CLASSES}

    for tile_path in sample:
        lbl_path = lbl_dir / (tile_path.stem + ".txt")
        img = draw_yolo_labels(tile_path, lbl_path)
        if img is None:
            continue

        # Count classes for the summary
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_counts[int(parts[0])] = class_counts.get(int(parts[0]), 0) + 1

        out_path = OUTPUT_DIR / tile_path.name
        cv2.imwrite(str(out_path), img)
        n_ok += 1

    print(f"\n✓ Saved {n_ok} verification images to:")
    print(f"  {OUTPUT_DIR}")
    print(f"\nClass counts in this sample:")
    for cid, count in class_counts.items():
        if count > 0:
            print(f"  {cid} {OPERATIONAL_CLASSES[cid]:18s}: {count}")
    print(f"\nOpen the folder and inspect a few — boxes should be tight on objects.")


if __name__ == "__main__":
    main()