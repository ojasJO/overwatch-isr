"""
config.py — Central configuration for the offline UAV ISR pipeline.
====================================================================
Single source of truth. Every tunable lives here; nothing downstream
hardcodes a value that belongs in this file.

Platform of record: Asteria A200-XT
  EO day camera · 3-axis gimbal · 1280x720 · dual GNSS · AMSL HUD overlay
  Pan 360° · Tilt spec -10°(up)..+90°(down) · Zoom 10x optical / 4x digital
  Wide FOV >= 50° · Narrow FOV <= 5° · records .avi (auto on ARM), .png snapshots
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORIES  (all resolved relative to this file, so the app is location-free)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path    = Path(__file__).resolve().parent
SRC_DIR: Path     = BASE_DIR / "src"
FOOTAGE_DIR: Path = BASE_DIR / "footage"   # drop .avi files here
HISTORY_DIR: Path = BASE_DIR / "history"   # geo-referenced frame overlays saved here
MODELS_DIR: Path  = BASE_DIR / "models"    # YOLO .pt / .engine weights
ARCHIVE_DIR: Path = BASE_DIR / "archive"   # obsolete Phase-1 files moved here on boot

HISTORY_MANIFEST: Path = HISTORY_DIR / "manifest.json"

for _d in (FOOTAGE_DIR, HISTORY_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
# Files from earlier proof-of-concept iterations that main.py archives on boot
# to keep the workspace pristine. Only moved if present; canonical files are
# never touched.
OBSOLETE_ARTIFACTS: List[str] = [
    "engine.py",
    "src/frame_grabber.py",
    "src/telemetry_schema.py",
    "src/telemetry.py",
    "src/asteria_parser.py",
    "dashboard",                 # old dashboard/ package (ZeroMQ SUB bridge)
]

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO INGESTION
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS: Tuple[str, ...] = (".avi", ".mp4", ".mov", ".mkv")
FRAME_STRIDE: int        = 3          # process every Nth frame (1 = every frame)
TRY_HW_ACCEL: bool       = True       # request GPU-assisted decode (CAP_PROP_HW_ACCELERATION)
RECONNECT_ON_EOF: bool   = False      # files end; don't loop

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA INTRINSICS  (live HFOV is read per-frame from the HUD; these are bounds)
# ─────────────────────────────────────────────────────────────────────────────
FRAME_WIDTH: int  = 1280
FRAME_HEIGHT: int = 720
HFOV_FALLBACK_DEG: float = 54.0       # used only if OCR cannot read HFOV
HFOV_MIN_DEG: float = 4.0             # A200-XT narrow bound
HFOV_MAX_DEG: float = 62.0            # A200-XT wide bound

# Raw HUD gimbal sanity bounds (pre-normalisation)
TILT_RAW_MIN: float = -100.0
TILT_RAW_MAX: float = 100.0
PAN_RAW_MIN: float  = -360.0
PAN_RAW_MAX: float  = 360.0

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION (YOLOv11)
# ─────────────────────────────────────────────────────────────────────────────
# Point at a TensorRT engine for max throughput on the 5060 Ti, e.g.
# "models/yolo11m.engine". If the .engine is missing, the .pt fallback is used.
MODEL_ENGINE: str = "models/yolo11m.engine"
MODEL_PT: str     = "models/yolo11m.pt"     # auto-downloads if absent
MODEL_FALLBACK_SIZE: str = "m"              # n|s|m|l|x for stock yolo11{SIZE}.pt

CONFIDENCE_THRESHOLD: float = 0.30
IOU_THRESHOLD: float        = 0.45
TARGET_CLASSES: List[int]   = [0, 1, 2, 3, 5, 7]   # person bicycle car motorcycle bus truck
IMG_SIZE: int               = 960
USE_HALF: bool              = True          # FP16 on CUDA

# Optional SAHI-style sliced inference for tiny top-down objects (slower).
USE_SAHI: bool       = False
SAHI_TILE: int       = 640
SAHI_OVERLAP: float  = 0.2
SAHI_MIN_HFOV: float = 20.0                 # only tile when zoomed out

CLASS_NAMES: Dict[int, str] = {
    0: "PERSON", 1: "BICYCLE", 2: "VEHICLE", 3: "MOTO", 5: "BUS", 7: "TRUCK",
}
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {   # BGR for OpenCV annotate
    "PERSON": (0, 210, 110), "BICYCLE": (255, 160, 0), "VEHICLE": (40, 110, 255),
    "MOTO": (210, 0, 200), "BUS": (0, 200, 220), "TRUCK": (0, 165, 255),
}
DEFAULT_COLOR: Tuple[int, int, int] = (180, 180, 180)

# ─────────────────────────────────────────────────────────────────────────────
# OCR (telemetry burned into the HUD overlay)
# ─────────────────────────────────────────────────────────────────────────────
OCR_USE_GPU: bool       = True
OCR_EVERY_N_FRAMES: int = 5             # OCR is expensive; cache between runs
OCR_CONFIDENCE_MIN: float = 0.35
OCR_DEBUG_SAVE_ROI: bool = False
DEBUG_PRINT_TELEMETRY: bool = False

# Plausibility gate: max change per second for slowly-varying HUD fields.
# A bigger jump between reads is treated as an OCR misread and rejected.
OCR_FIELD_MAX_RATE: Dict[str, float] = {
    "agl": 30.0, "amsl": 30.0, "hfov": 40.0,
    "tilt": 60.0, "pan": 180.0, "heading": 180.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# GEOLOCATION + DRIFT CONTROL
# ─────────────────────────────────────────────────────────────────────────────
ALTITUDE_REF: str = "agl"               # ray-cast uses height-above-ground

MIN_DEPRESSION_DEG: float = 3.0         # refuse near-horizon geometry
MAX_RANGE_M: float        = 8000.0      # reject absurd ground ranges

# Platform isolation: the airframe is static, so its plotted position is held
# steady with a slow low-pass; gimbal/zoom changes must NOT move the station.
PLATFORM_EMA_ALPHA: float = 0.05        # smaller = stiffer/steadier station point

# TGT-anchor bias removal (HUD provides the drone's own target solution).
USE_TGT_ANCHOR: bool  = True
BIAS_LP_ALPHA: float  = 0.15
BIAS_MAX_M: float     = 250.0

# Per-target temporal smoothing on the OUTPUT ground coordinate (decoupled from
# raw telemetry jitter). Constant-position Kalman in local ENU metres.
TRACK_ENABLE: bool         = True
TRACK_ASSOC_M: float       = 25.0
TRACK_MAX_AGE_FRAMES: int  = 45
TRACK_Q_M: float           = 1.5        # process noise (m): how fast a target may truly move
TRACK_MIN_R_M: float       = 2.0        # measurement-noise floor (m)

# 1-sigma error inputs for analytic error propagation -> per-target accuracy (m).
SIGMA_GNSS_M: float   = 4.0
SIGMA_TILT_DEG: float = 1.0
SIGMA_PAN_DEG: float  = 1.0
SIGMA_HDG_DEG: float  = 2.0
SIGMA_AGL_FRAC: float = 0.05

VALIDATION_PIXEL: Tuple[int, int] = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
VALIDATION_THRESHOLD_M: float     = 200.0

MGRS_PRECISION: int = 5                 # 5 -> 1 m grid (digits per easting/northing)

# ─────────────────────────────────────────────────────────────────────────────
# TERRAIN (DEM)  — Copernicus GLO-30 GeoTIFF for true 3D ground intersection
# ─────────────────────────────────────────────────────────────────────────────
USE_DEM: bool          = True
DEM_PATH: Path         = BASE_DIR / "data" / "dem" / "fazilkacantt_dem.tif"
DEM_PATH.parent.mkdir(parents=True, exist_ok=True)
DEM_MARCH_STEP_M: float = 5.0           # coarse ray-march step (bisection refines)
DEM_BISECT_ITERS: int   = 25            # bisection iterations (-> sub-cm along-ray)

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY (geo-referenced frame overlays for the mosaic + time scrubber)
# ─────────────────────────────────────────────────────────────────────────────
HISTORY_SAVE_MIN_INTERVAL_S: float = 2.0    # min seconds between saved overlays
HISTORY_SAVE_ON_DETECTION: bool    = True   # always save a frame that has contacts
HISTORY_JPEG_QUALITY: int          = 80
HISTORY_MAX_FRAMES: int            = 600    # rolling cap; oldest pruned beyond this
HISTORY_OVERLAY_MAX_W: int         = 960    # downscale saved overlays for the browser

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HOST: str   = "0.0.0.0"
DASHBOARD_PORT: int   = 5000
LIVE_JPEG_QUALITY: int = 70
LIVE_JPEG_MAX_W: int   = 900
LIVE_EMIT_EVERY_N: int = 2              # throttle live viewport frames to the browser
MAX_FEED_DETECTIONS: int = 60

SYSTEM_NAME: str          = "UAV-ISR // TACTICAL"
CLASSIFICATION_LABEL: str = "UNCLASSIFIED · FOR OFFICIAL USE ONLY"

# ─────────────────────────────────────────────────────────────────────────────
# ORTHOMOSAIC STITCHER  (src/stitcher.py)
# ─────────────────────────────────────────────────────────────────────────────
STITCH_USE_GPU: bool          = True     # use cv2.cuda IF the OpenCV build has it
STITCH_GSD_M: float           = 0.15     # orthomosaic ground sample distance (m/px)
STITCH_MIN_AGL_M: float       = 50.0     # below this AGL: skip (too low for a macro map)
STITCH_MAX_ZOOM: float        = 1.0      # above this optical zoom: skip (not a wide map)
STITCH_MIN_DEPRESSION_DEG: float = 45.0  # below this gimbal depression: skip (2D ortho invalid)
STITCH_ORB_FEATURES: int      = 4000
STITCH_MATCH_RATIO: float     = 0.75     # Lowe ratio test
STITCH_MIN_MATCHES: int       = 18       # min inliers to trust a feature refinement
STITCH_RANSAC_THRESH: float   = 5.0      # px reprojection threshold for RANSAC
STITCH_REFINE_WITH_FEATURES: bool = True # snap each tile to the mosaic via ORB+RANSAC
STITCH_REFINE_MAX_SHIFT_M: float = 15.0  # reject feature corrections larger than this
STITCH_FEATHER_PX: int        = 40       # blend feather width
STITCH_MAX_CANVAS_PX: int     = 16000    # hard cap on either mosaic dimension
STITCH_MASK_HUD: bool         = True     # exclude burned-in HUD text from the mosaic
# fractional (y0, y1, x0, x1) bands of the frame that carry HUD overlay text
STITCH_HUD_BANDS: List[Tuple[float, float, float, float]] = [
    (0.00, 0.13, 0.00, 1.00),            # top status band
    (0.84, 1.00, 0.00, 1.00),            # bottom telemetry band
    (0.56, 0.78, 0.68, 1.00),            # mid-right gimbal block
]
MOSAIC_OUTPUT: Path           = HISTORY_DIR / "orthomosaic.jpg"
STITCH_MOSAIC_REFRESH_S: float = 5.0     # how often to rewrite + push the composite mosaic to the UI