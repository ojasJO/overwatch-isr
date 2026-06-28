"""
integrate_model.py  —  Drop the fine-tuned xView model into your pipeline
=========================================================================
After training completes, this script:
  1. Copies best.pt to ~/uav_pipeline/models/yolo11_xview.pt
  2. Shows you exactly which lines to update in src/detector.py
  3. Shows you exactly which lines to update in config.py

Run AFTER train_yolo.py:
    python3 scripts/integrate_model.py
"""

import shutil
import sys
from pathlib import Path

HOME        = Path.home()
RUNS_DIR    = HOME / "uav_pipeline" / "runs"
RUN_NAME    = "xview_yolov11s_aerial"
SRC_WEIGHTS = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
DST_DIR     = HOME / "uav_pipeline" / "models"
DST_WEIGHTS = DST_DIR / "yolo11_xview.pt"


def main() -> None:
    if not SRC_WEIGHTS.exists():
        print(f"ERROR: Trained weights not found at:")
        print(f"  {SRC_WEIGHTS}")
        print(f"\nDid training finish? Check {RUNS_DIR / RUN_NAME}")
        sys.exit(1)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    size_mb = SRC_WEIGHTS.stat().st_size / 1e6
    print(f"Copying {size_mb:.1f} MB weights …")
    shutil.copy2(SRC_WEIGHTS, DST_WEIGHTS)
    print(f"✓ Copied to {DST_WEIGHTS}")

    # Print integration instructions
    print("\n" + "=" * 70)
    print("INTEGRATION INSTRUCTIONS")
    print("=" * 70)
    print("\n1.  In ~/uav_pipeline/config.py, ADD or REPLACE:")
    print()
    print(f"    MODEL_PATH           = '{DST_WEIGHTS}'")
    print(f"    CONFIDENCE_THRESHOLD = 0.35    # xView model is more confident")
    print(f"    TARGET_CLASSES       = [0, 1, 2, 3, 4, 5]  # all 6 operational classes")
    print()
    print("2.  In ~/uav_pipeline/src/detector.py, find this line in __init__:")
    print()
    print(f"      model_file = f\"yolo11{{config.MODEL_SIZE}}.pt\"")
    print()
    print("    REPLACE with:")
    print()
    print(f"      model_file = config.MODEL_PATH if hasattr(config, 'MODEL_PATH') \\")
    print(f"                   else f\"yolo11{{config.MODEL_SIZE}}.pt\"")
    print()
    print("3.  Update CLASS_CONFIG in src/detector.py — replace the COCO dict with:")
    print()
    print("    CLASS_CONFIG = {")
    print("        'small_vehicle':  {'colour': (80, 200, 80),  'label': 'VEHICLE'},")
    print("        'large_vehicle':  {'colour': (0, 140, 255),  'label': 'LARGE VEH'},")
    print("        'aircraft':       {'colour': (255, 140, 0),  'label': 'AIRCRAFT'},")
    print("        'vessel':         {'colour': (180, 80, 255), 'label': 'VESSEL'},")
    print("        'construction':   {'colour': (40, 200, 220), 'label': 'CONSTR'},")
    print("        'building':       {'colour': (200, 200, 200),'label': 'BUILDING'},")
    print("    }")
    print()
    print("4.  Run your pipeline as normal:")
    print()
    print("    python3 main.py")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()