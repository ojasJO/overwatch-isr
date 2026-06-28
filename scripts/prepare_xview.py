"""
prepare_xview.py  —  xView → YOLO Format with Tiling
=====================================================

Pipeline:
  1.  Read xView_train.geojson
  2.  Group annotations by image
  3.  Split images 80/20 train/val (deterministic seed)
  4.  For each image:
        - Load the .tif
        - Slice into 1024×1024 tiles with 200px overlap
        - For each tile, find boxes inside and rewrite YOLO format
        - Save tile as .jpg (smaller than .tif, YOLO-friendly)
        - Save matching .txt label
  5.  Drop tiles with zero objects (with small probability of keeping
      empty tiles as negative examples — 10%)
  6.  Generate data.yaml for YOLO training

Output structure:
  ~/uav_pipeline/datasets/xview_yolo/
      ├── images/
      │   ├── train/  (~50k jpgs)
      │   └── val/    (~12k jpgs)
      ├── labels/
      │   ├── train/  (~50k txts)
      │   └── val/    (~12k txts)
      └── data.yaml

Run:
    cd ~/uav_pipeline
    source venv/bin/activate
    python3 scripts/prepare_xview.py
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # xView tifs are huge

# Local
sys.path.insert(0, str(Path(__file__).parent))
from xview_classes import xview_to_op, OPERATIONAL_CLASSES

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOME           = Path.home()
DATASETS_ROOT  = HOME / "uav_pipeline" / "datasets"
XVIEW_RAW      = DATASETS_ROOT / "xview_raw"
GEOJSON_PATH   = XVIEW_RAW / "xView_train.geojson"
IMG_DIR        = XVIEW_RAW / "train"        # where the .tif images live

OUTPUT_ROOT    = DATASETS_ROOT / "xview_yolo"

# Tiling parameters
TILE_SIZE      = 1024     # YOLO will train at this resolution
TILE_OVERLAP   = 200      # overlap between tiles (catches objects on borders)
MIN_BOX_AREA   = 50       # drop boxes smaller than this in tile coords (pixels²)
MIN_VISIBILITY = 0.30     # drop boxes if <30% visible in tile after cropping
KEEP_EMPTY_PROB= 0.05     # keep 5% of empty tiles as negative examples
JPEG_QUALITY   = 92       # tile JPEG quality (92 = visually lossless, smaller file)

# Train/val split
VAL_FRACTION   = 0.20
SEED           = 42

# Speed / logging
PROGRESS_EVERY = 25       # print progress every N images
DRY_RUN        = False    # True = process only first 5 images (sanity check)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_geojson(path: Path) -> dict[str, list[dict]]:
    """
    Parse xView_train.geojson and group annotations by image filename.

    Returns
    -------
    dict
        { "1234.tif": [ {bbox: [x1,y1,x2,y2], class: 17}, ... ], ... }
    """
    log(f"Loading {path.name} ({path.stat().st_size / 1e6:.0f} MB) …")
    with open(path, "r") as f:
        gj = json.load(f)

    log(f"Found {len(gj['features'])} annotation features")
    by_image: dict[str, list[dict]] = defaultdict(list)
    skipped_unmapped = 0
    skipped_bad_bbox = 0

    for feat in gj["features"]:
        props  = feat["properties"]
        img_id = props["image_id"]                # e.g. "1234.tif"
        xv_cls = int(props["type_id"])
        bbox   = props["bounds_imcoords"]         # "x1,y1,x2,y2"

        op_cls = xview_to_op(xv_cls)
        if op_cls is None:
            skipped_unmapped += 1
            continue

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox.split(",")]
            if x2 <= x1 or y2 <= y1:
                skipped_bad_bbox += 1
                continue
        except (ValueError, AttributeError):
            skipped_bad_bbox += 1
            continue

        by_image[img_id].append({
            "bbox":  (x1, y1, x2, y2),
            "class": op_cls,
        })

    log(f"  Grouped into {len(by_image)} images")
    log(f"  Skipped {skipped_unmapped} unmapped class annotations")
    log(f"  Skipped {skipped_bad_bbox} bad-bbox annotations")
    total_boxes = sum(len(v) for v in by_image.values())
    log(f"  Total usable boxes: {total_boxes:,}")
    return dict(by_image)


def tile_image(
    img_path: Path,
    boxes:    list[dict],
    out_img_dir: Path,
    out_lbl_dir: Path,
    rng:      random.Random,
) -> tuple[int, int, int]:
    """
    Slice one xView .tif into overlapping tiles.
    For each tile, write YOLO-format labels for boxes contained in it.

    Returns
    -------
    (tiles_with_boxes, tiles_empty_kept, total_boxes_written)
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        log(f"  ERROR opening {img_path.name}: {e}")
        return 0, 0, 0

    W, H = img.size
    stride = TILE_SIZE - TILE_OVERLAP
    stem   = img_path.stem  # e.g. "1234"

    n_with = 0
    n_empty = 0
    n_boxes = 0

    # Generate tile origins
    xs = list(range(0, max(1, W - TILE_OVERLAP), stride))
    ys = list(range(0, max(1, H - TILE_OVERLAP), stride))
    # Ensure last tile reaches image edge
    if xs and xs[-1] + TILE_SIZE < W:
        xs.append(W - TILE_SIZE)
    if ys and ys[-1] + TILE_SIZE < H:
        ys.append(H - TILE_SIZE)

    for tx in xs:
        for ty in ys:
            tx2 = min(tx + TILE_SIZE, W)
            ty2 = min(ty + TILE_SIZE, H)
            tw  = tx2 - tx
            th  = ty2 - ty
            if tw < TILE_SIZE * 0.5 or th < TILE_SIZE * 0.5:
                continue   # skip tiny edge tiles

            # Find boxes inside this tile
            tile_labels = []
            for b in boxes:
                bx1, by1, bx2, by2 = b["bbox"]

                # Intersection with tile
                ix1 = max(bx1, tx)
                iy1 = max(by1, ty)
                ix2 = min(bx2, tx2)
                iy2 = min(by2, ty2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter_area = (ix2 - ix1) * (iy2 - iy1)
                box_area   = (bx2 - bx1) * (by2 - by1)
                if box_area <= 0:
                    continue

                visibility = inter_area / box_area
                if visibility < MIN_VISIBILITY:
                    continue
                if inter_area < MIN_BOX_AREA:
                    continue

                # Convert clipped box to tile-relative pixel coords
                lx1 = ix1 - tx
                ly1 = iy1 - ty
                lx2 = ix2 - tx
                ly2 = iy2 - ty

# YOLO format: normalized cx, cy, w, h
                # CRITICAL: normalize by TILE_SIZE (the saved/padded image size),
                # NOT by tw/th (the cropped region size). Edge tiles are padded
                # to TILE_SIZE × TILE_SIZE — the boxes live in the unpadded
                # top-left region, so dividing by TILE_SIZE gives correct coords.
                cx = ((lx1 + lx2) / 2.0) / TILE_SIZE
                cy = ((ly1 + ly2) / 2.0) / TILE_SIZE
                bw = (lx2 - lx1) / TILE_SIZE
                bh = (ly2 - ly1) / TILE_SIZE

                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))

                if bw < 0.005 or bh < 0.005:  # degenerate
                    continue

                tile_labels.append(f"{b['class']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Decide whether to save this tile
            keep = bool(tile_labels) or (rng.random() < KEEP_EMPTY_PROB)
            if not keep:
                continue

            # Crop & save
            tile = img.crop((tx, ty, tx2, ty2))
            if (tw, th) != (TILE_SIZE, TILE_SIZE):
                # Pad to TILE_SIZE × TILE_SIZE with black (right/bottom)
                padded = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile = padded

            tile_name = f"{stem}_x{tx}_y{ty}"
            tile.save(out_img_dir / f"{tile_name}.jpg",
                      "JPEG", quality=JPEG_QUALITY, optimize=True)

            with open(out_lbl_dir / f"{tile_name}.txt", "w") as f:
                f.write("\n".join(tile_labels))

            if tile_labels:
                n_with += 1
                n_boxes += len(tile_labels)
            else:
                n_empty += 1

    img.close()
    return n_with, n_empty, n_boxes


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    rng = random.Random(SEED)

    # 1. Validate inputs
    if not GEOJSON_PATH.exists():
        log(f"ERROR: GeoJSON not found at {GEOJSON_PATH}")
        sys.exit(1)
    if not IMG_DIR.exists():
        log(f"ERROR: Image directory not found at {IMG_DIR}")
        sys.exit(1)

    # 2. Prepare output dirs
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # 3. Load annotations
    by_image = load_geojson(GEOJSON_PATH)

    # 4. Filter to only images that actually exist on disk
    available_tifs = {p.name for p in IMG_DIR.glob("*.tif")}
    log(f"Found {len(available_tifs)} .tif files on disk")
    by_image = {k: v for k, v in by_image.items() if k in available_tifs}
    log(f"After matching to disk: {len(by_image)} annotated images")

    if DRY_RUN:
        keys = list(by_image.keys())[:5]
        by_image = {k: by_image[k] for k in keys}
        log(f"DRY RUN: processing only {len(by_image)} images")

    # 5. Train/val split (by image, not by box)
    image_ids = sorted(by_image.keys())
    rng.shuffle(image_ids)
    n_val = int(len(image_ids) * VAL_FRACTION)
    val_set   = set(image_ids[:n_val])
    train_set = set(image_ids[n_val:])
    log(f"Split: {len(train_set)} train images, {len(val_set)} val images")

    # 6. Process each image: tile + write labels
    stats = {"train": [0, 0, 0], "val": [0, 0, 0]}  # [with_boxes, empty, boxes]
    t0 = time.time()

    for i, img_name in enumerate(image_ids, 1):
        split = "val" if img_name in val_set else "train"
        img_path = IMG_DIR / img_name
        out_img_dir = OUTPUT_ROOT / "images" / split
        out_lbl_dir = OUTPUT_ROOT / "labels" / split

        nw, ne, nb = tile_image(
            img_path, by_image[img_name],
            out_img_dir, out_lbl_dir, rng,
        )
        stats[split][0] += nw
        stats[split][1] += ne
        stats[split][2] += nb

        if i % PROGRESS_EVERY == 0 or i == len(image_ids):
            elapsed = time.time() - t0
            rate = i / elapsed
            eta  = (len(image_ids) - i) / rate if rate > 0 else 0
            log(f"[{i:4d}/{len(image_ids)}]  "
                f"train: {stats['train'][0]} tiles ({stats['train'][2]} boxes)  "
                f"val: {stats['val'][0]} tiles ({stats['val'][2]} boxes)  "
                f"rate: {rate:.1f} img/s  ETA: {eta/60:.1f} min")

    # 7. Write data.yaml
    yaml_path = OUTPUT_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"# xView fine-tuning dataset for YOLOv11\n")
        f.write(f"# Generated by prepare_xview.py\n\n")
        f.write(f"path: {OUTPUT_ROOT.absolute()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"nc: {len(OPERATIONAL_CLASSES)}\n")
        f.write(f"names:\n")
        for cid in sorted(OPERATIONAL_CLASSES.keys()):
            f.write(f"  {cid}: {OPERATIONAL_CLASSES[cid]}\n")

    log("=" * 70)
    log("DONE")
    log(f"  Train: {stats['train'][0]} tiles with boxes, "
        f"{stats['train'][1]} empty kept, {stats['train'][2]:,} boxes")
    log(f"  Val:   {stats['val'][0]} tiles with boxes, "
        f"{stats['val'][1]} empty kept, {stats['val'][2]:,} boxes")
    log(f"  Output: {OUTPUT_ROOT}")
    log(f"  data.yaml: {yaml_path}")
    log("=" * 70)


if __name__ == "__main__":
    main()