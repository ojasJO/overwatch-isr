#!/bin/bash
# =============================================================================
# restart_training.sh — Clean up + restart YOLOv11 fine-tuning on xView
# =============================================================================
# What this script does:
#   1. DELETES junk files (1.png, 2.avi, ocr_overlay.py, duplicate weights)
#   2. PROTECTS your dataset, pipeline code, and any existing trained models
#   3. Wipes the previous run directory (which contains partial training)
#   4. Updates train_yolo.py with proven crash-resistant settings
#   5. Kicks off fresh training in tmux
#
# This script is SAFE to run multiple times.
# =============================================================================

set -e   # exit on any error

UAV_DIR="$HOME/uav_pipeline"
cd "$UAV_DIR" || { echo "ERROR: $UAV_DIR not found"; exit 1; }

echo "═══════════════════════════════════════════════════════════════"
echo "  RESTART TRAINING — clean slate"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── 1. Show current state ───────────────────────────────────────────────────
echo "[1/6] Current state:"
echo "  Dataset tiles:  $(ls datasets/xview_yolo/images/train 2>/dev/null | wc -l) train, $(ls datasets/xview_yolo/images/val 2>/dev/null | wc -l) val"
echo "  Disk free:      $(df -h $HOME | awk 'NR==2 {print $4}')"
if [ -d runs/xview_yolov11s_aerial ]; then
    echo "  Previous run:   EXISTS (will be deleted)"
fi
if [ -f models/yolo11_xview.pt ]; then
    echo "  Previous model: EXISTS (will be backed up to models/yolo11_xview_old.pt)"
fi
echo ""

# ── 2. Delete junk files (safe deletes) ─────────────────────────────────────
echo "[2/6] Deleting junk files..."
JUNK=(
    "1.png" "2.avi" "annotated_1.png" "annotated_boxes.png"
    "ocr_overlay.py"
    "main_1.py"
    "yolo11n.pt" "yolo26n.pt"     # downloaded base models, will re-download if needed
)
for f in "${JUNK[@]}"; do
    if [ -e "$f" ]; then
        rm -v "$f"
    fi
done

# Clean Python cache
find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
echo "  Cleaned __pycache__"
echo ""

# ── 3. Back up existing trained model ───────────────────────────────────────
echo "[3/6] Backing up existing model (if any)..."
mkdir -p models
if [ -f models/yolo11_xview.pt ]; then
    mv -v models/yolo11_xview.pt models/yolo11_xview_old.pt
    echo "  ✓ Old model preserved as models/yolo11_xview_old.pt"
else
    echo "  No existing model to back up"
fi
echo ""

# ── 4. Wipe previous training run ───────────────────────────────────────────
echo "[4/6] Wiping previous training run..."
if [ -d runs/xview_yolov11s_aerial ]; then
    rm -rf runs/xview_yolov11s_aerial
    echo "  ✓ Deleted runs/xview_yolov11s_aerial"
fi
if [ -d runs/detect ]; then
    rm -rf runs/detect       # validation results from old run
    echo "  ✓ Deleted runs/detect"
fi
echo ""

# ── 5. Verify protected files still exist ───────────────────────────────────
echo "[5/6] Verifying protected files..."
REQUIRED=(
    "config.py"
    "main.py"
    "src/detector.py"
    "src/geolocator.py"
    "src/asteria_parser.py"
    "src/telemetry.py"
    "dashboard/app.py"
    "scripts/train_yolo.py"
    "scripts/prepare_xview.py"
    "scripts/xview_classes.py"
    "datasets/xview_yolo/data.yaml"
)
MISSING=0
for f in "${REQUIRED[@]}"; do
    if [ ! -e "$f" ]; then
        echo "  ✗ MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING required file(s) missing. Cannot proceed."
    exit 1
fi
echo "  ✓ All protected files present"
echo ""

# ── 6. Check GPU is available ───────────────────────────────────────────────
echo "[6/6] Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ✗ nvidia-smi not found"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits)
GPU_FREE_MB=$(echo "$GPU_INFO" | awk -F',' '{print $2}' | tr -d ' ')
echo "  GPU: $(echo "$GPU_INFO" | awk -F',' '{print $1}')"
echo "  Free VRAM: ${GPU_FREE_MB} MB"

if [ "$GPU_FREE_MB" -lt 14000 ]; then
    echo ""
    echo "WARNING: Less than 14 GB free VRAM. Other processes using GPU?"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo ""
    read -p "Continue anyway? (y/N): " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi
echo ""

# ── Done with cleanup, ready to train ───────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  READY TO TRAIN"
echo "  Settings: YOLOv11s @ 768px, batch=4, workers=2, 80 epochs"
echo "  Expected: 8-12 hours, mAP50 ~0.55-0.65"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Make sure tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    sudo apt install -y tmux
fi

# Kill any existing 'train' session
tmux kill-session -t train 2>/dev/null || true

echo "Starting training in tmux session 'train'..."
echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  After this prints, training will start in a tmux session."
echo ""
echo "  To watch progress:    tmux attach -t train"
echo "  To detach (keep running):  Ctrl+B then D"
echo "  To check from another terminal:  tmux attach -t train"
echo "  To kill training:     tmux kill-session -t train"
echo "──────────────────────────────────────────────────────────────"
echo ""
sleep 2

# Launch training in tmux
tmux new-session -d -s train -c "$UAV_DIR" \
    "source venv/bin/activate && python3 scripts/train_yolo.py 2>&1 | tee runs/training_log.txt"

echo "✓ Training started in background tmux session"
echo ""
echo "Attach now to watch live:   tmux attach -t train"
echo "Or just leave it running."