"""
src/video_processor.py — Frame ingestion + HUD OCR + history persistence.
=========================================================================
Offline. Opens a recorded .avi, streams frames (with HW-accelerated decode
when the build supports it), reads the burned-in telemetry overlay via
EasyOCR, and persists high-value geo-referenced frames to ./history for the
dashboard's mosaic-overlap + time-scrubber.

Telemetry robustness:
  * RAW values only — a legitimate 0 (heading due-north, pan 0) is preserved,
    never discarded by a falsy `or`.
  * Range-bounded — impossible reads are dropped.
  * Plausibility-gated + last-known-good — an OCR digit-misread that implies a
    physically impossible jump between frames is rejected and the previous good
    value is held, with a staleness clock.
"""
from __future__ import annotations
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger("VideoProcessor")

# EasyOCR is heavy; load once, lazily.
_OCR_READER = None


def _get_reader():
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            logger.info("Initialising EasyOCR (first call ~3-5s) …")
            _OCR_READER = easyocr.Reader(["en"], gpu=config.OCR_USE_GPU, verbose=False)
        except Exception as exc:
            logger.error(f"EasyOCR unavailable ({exc}); telemetry OCR disabled")
            _OCR_READER = False
    return _OCR_READER


# Regions of the HUD to OCR (fractional y0, y1, x0, x1).
# Mapped from actual Asteria A200-XT footage (1660x937 screenshot, scales to 1280x720).
_REGIONS = {
    "top_left":     (0.00, 0.12, 0.00, 0.20),   # A2X-A128, UAV Mode, UTC
    "top_center":   (0.00, 0.09, 0.38, 0.62),    # FLT timer
    "mid_right":    (0.56, 0.78, 0.68, 1.00),    # ZOOM(O/D), PAN, TILT, HFOV  ← CRITICAL
    "bottom_left":  (0.85, 1.00, 0.10, 0.62),    # UAV lat/lon, SPEED, DTH, AMSL, HDG, BRG, AGL, AHL
    "bottom_right": (0.85, 1.00, 0.56, 1.00),    # TGT lat/lon, AMSL, SR
}

_RE = {
    "lat":      re.compile(r'UAV\s+([\d]+\.[\d]+)\s*N', re.I),
    "lon":      re.compile(r'UAV\s+[\d.]+\s*N\s+([\d]+\.[\d]+)\s*E', re.I),
    "speed":    re.compile(r'SPEED\s+(\d+)\s*kmph', re.I),
    "dth":      re.compile(r'DTH\s+([\d.]+)\s*km', re.I),
    "amsl":     re.compile(r'AMSL\s+(\d+)\s*m\b', re.I),
    "hdg":      re.compile(r'HDG\s+(\d+)', re.I),
    "brg":      re.compile(r'BRG[:\s]+(\d+)', re.I),
    "agl":      re.compile(r'AGL\s+(\d+)\s*m', re.I),
    "ahl":      re.compile(r'AHL\s+(\d+)\s*m', re.I),
    "tgt_lat":  re.compile(r'TGT[:\s]+([\d]+\.[\d]+)\s*N', re.I),
    "tgt_lon":  re.compile(r'TGT[:\s]+[\d.]+\s*N\s+([\d]+\.[\d]+)\s*E', re.I),
    "sr":       re.compile(r'SR[:\s]*0*(\d+)\s*m', re.I),
    "pan":      re.compile(r'PAN[:\s]+([-]?\d+)', re.I),
    "tilt":     re.compile(r'TILT[:\s]+([-]?\d+)', re.I),
    "hfov":     re.compile(r'HFOV[:\s]+([\d.]+)', re.I),
    "zoom_o":   re.compile(r'ZOOM\(O\)[:\s]+([\d.]+)', re.I),
    "zoom_d":   re.compile(r'ZOOM\(D\)[:\s]+([\d.]+)', re.I),
    "drone_id": re.compile(r'(A\d+X[-_]A\d+)', re.I),
    "mode":     re.compile(r'UAV\s+Mode[:\s]+(\w+)', re.I),
    "utc":      re.compile(r'UTC[:\s]+(\d{2}[A-Z]{3}\d{2}\s+\d{2}:\d{2}:\d{2})', re.I),
    "flt":      re.compile(r'FLT[:\s]+(\d{2}:\d{2}:\d{2})', re.I),
}

_BOUNDS = {
    "amsl": (-500, 12000), "agl": (0, 12000), "ahl": (0, 12000),
    "heading": (0, 360), "brg": (0, 360), "speed_kmph": (0, 400),
    "tilt": (config.TILT_RAW_MIN, config.TILT_RAW_MAX),
    "pan": (config.PAN_RAW_MIN, config.PAN_RAW_MAX),
    "hfov": (config.HFOV_MIN_DEG, config.HFOV_MAX_DEG),
    "sr": (0, 20000), "zoom_o": (1, 10), "zoom_d": (1, 4),
}


class VideoProcessor:
    """Wraps one video file: iterate frames + OCR the overlay."""

    def __init__(self, video_path: Path):
        self.path = Path(video_path)
        self.cap = self._open(self.path)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or config.FRAME_WIDTH)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.FRAME_HEIGHT)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._good: Dict[str, Tuple[float, float]] = {}   # field -> (value, t_seen)
        self._ocr_cache: Optional[Dict] = None
        self._ocr_ok = 0
        self._ocr_fail = 0
        logger.info(f"Opened {self.path.name} · {self.width}x{self.height} · "
                    f"{self.fps:.1f} fps · {self.frame_count} frames")

    # ── open with optional HW-accelerated decode ──────────────────────────────
    @staticmethod
    def _open(path: Path) -> cv2.VideoCapture:
        cap = None
        if config.TRY_HW_ACCEL:
            try:
                params = [int(cv2.CAP_PROP_HW_ACCELERATION), int(cv2.VIDEO_ACCELERATION_ANY)]
                cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, params)
                if not cap.isOpened():
                    cap = None
            except Exception as exc:
                logger.debug(f"HW-accel open failed ({exc}); falling back")
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        return cap

    # ── frame iterator ─────────────────────────────────────────────────────--
    def frames(self, stride: int = None) -> Iterator[Tuple[int, float, np.ndarray]]:
        stride = stride or config.FRAME_STRIDE
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            if idx % stride == 0:
                t_sec = idx / self.fps if self.fps > 0 else idx / 25.0
                yield idx, t_sec, frame
            idx += 1
        logger.info(f"{self.path.name}: end of stream at frame {idx}")

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    @property
    def ocr_stats(self) -> Dict[str, int]:
        return {"ok": self._ocr_ok, "fail": self._ocr_fail}

    # ── OCR the overlay (throttled, gated, last-known-good) ────────────────────
    def parse_overlay(self, frame: np.ndarray, frame_idx: int, t_now: float) -> Dict:
        if frame_idx % config.OCR_EVERY_N_FRAMES != 0 and self._ocr_cache is not None:
            return self._ocr_cache

        reader = _get_reader()
        raw_text = self._ocr_text(frame, reader) if reader else ""
        parsed = self._parse_text(raw_text, t_now) if raw_text else None
        gated = self._gate(parsed, t_now)

        if parsed and parsed.get("lat") is not None:
            self._ocr_ok += 1
        else:
            self._ocr_fail += 1
        self._ocr_cache = gated
        if config.DEBUG_PRINT_TELEMETRY and gated.get("lat") is not None:
            logger.info(f"[OCR] {gated.get('drone_id')} "
                        f"({gated.get('lat'):.5f},{gated.get('lon'):.5f}) "
                        f"AGL={gated.get('agl')} TILT={gated.get('tilt')} HFOV={gated.get('hfov')}")
        return gated

    def _ocr_text(self, frame: np.ndarray, reader) -> str:
        h, w = frame.shape[:2]
        parts: List[str] = []
        for name, (y0, y1, x0, x1) in _REGIONS.items():
            roi = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=20)
            _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            if config.OCR_DEBUG_SAVE_ROI:
                cv2.imwrite(f"/tmp/uav_ocr_{name}.png", bw)
            try:
                res = reader.readtext(bw, detail=1, paragraph=False,
                                      text_threshold=config.OCR_CONFIDENCE_MIN)
            except Exception as exc:
                logger.debug(f"OCR readtext failed on {name}: {exc}")
                res = []
            parts.append(" ".join(r[1] for r in res if r[2] >= config.OCR_CONFIDENCE_MIN))
        return " | ".join(parts)

    @staticmethod
    def _parse_text(text: str, ts: float) -> Optional[Dict]:
        def num(key: str) -> Optional[float]:
            m = _RE[key].search(text)
            if not m:
                return None
            try:
                v = float(m.group(1))
            except ValueError:
                return None
            lo_hi = _BOUNDS.get(key)
            if lo_hi and not (lo_hi[0] <= v <= lo_hi[1]):
                return None
            return v

        def txt(key: str) -> Optional[str]:
            m = _RE[key].search(text)
            return m.group(1) if m else None

        lat, lon = num("lat"), num("lon")
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        return {
            "lat": lat, "lon": lon, "amsl": num("amsl"), "agl": num("agl"), "ahl": num("ahl"),
            "heading": num("hdg"), "brg": num("brg"), "speed_kmph": num("speed"),
            "dth_km": num("dth"), "tilt": num("tilt"), "pan": num("pan"), "hfov": num("hfov"),
            "zoom_o": num("zoom_o"), "zoom_d": num("zoom_d"),
            "tgt_lat": num("tgt_lat"), "tgt_lon": num("tgt_lon"), "sr": num("sr"),
            "drone_id": txt("drone_id"), "mode": txt("mode"), "utc": txt("utc"), "flt": txt("flt"),
            "timestamp": ts, "source": "asteria_ocr",
        }

    def _gate(self, parsed: Optional[Dict], now: float) -> Dict:
        """Plausibility gate + last-known-good hold."""
        parsed = parsed or {}
        numeric = ("lat", "lon", "amsl", "agl", "ahl", "heading", "brg", "speed_kmph",
                   "dth_km", "tilt", "pan", "hfov", "zoom_o", "zoom_d",
                   "tgt_lat", "tgt_lon", "sr")
        out: Dict = {}
        for key in numeric:
            v = parsed.get(key)
            if v is not None and self._plausible(key, v, now):
                self._good[key] = (v, now)
            out[key] = self._good.get(key, (None, 0.0))[0]
        for key in ("drone_id", "mode", "utc", "flt", "source"):
            if parsed.get(key):
                self._good[key] = (parsed[key], now)
            out[key] = self._good.get(key, ("", 0.0))[0]
        out["timestamp"] = parsed.get("timestamp", now)
        return out

    def _plausible(self, key: str, value: float, now: float) -> bool:
        rate = config.OCR_FIELD_MAX_RATE.get(key)
        if rate is not None and key in self._good:
            prev, t_prev = self._good[key]
            dt = max(1e-3, now - t_prev)
            if abs(value - prev) > rate * dt + rate:
                return False
        return True


# ════════════════════════════════════════════════════════════════════════════
# History writer — geo-referenced frame overlays + manifest
# ════════════════════════════════════════════════════════════════════════════
class HistoryWriter:
    """Persists high-value frames + their N/S/E/W bounds for the UI mosaic."""

    def __init__(self):
        config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict] = self._load()
        self._last_save_t = 0.0
        logger.info(f"HistoryWriter ready · {len(self.entries)} existing overlays")

    def _load(self) -> List[Dict]:
        if config.HISTORY_MANIFEST.exists():
            try:
                return json.loads(config.HISTORY_MANIFEST.read_text())
            except Exception as exc:
                logger.warning(f"could not read history manifest ({exc}); starting fresh")
        return []

    def _flush(self) -> None:
        try:
            config.HISTORY_MANIFEST.write_text(json.dumps(self.entries, indent=0))
        except Exception as exc:
            logger.error(f"failed to write history manifest: {exc}")

    def should_save(self, t_wall: float, has_detection: bool) -> bool:
        if has_detection and config.HISTORY_SAVE_ON_DETECTION:
            interval = 0.0
        else:
            interval = config.HISTORY_SAVE_MIN_INTERVAL_S
        return (t_wall - self._last_save_t) >= interval

    def save(self, frame_bgr: np.ndarray, geo_bbox: Dict[str, float],
             meta: Dict, t_wall: float) -> Optional[Dict]:
        if geo_bbox is None:
            return None
        try:
            img = frame_bgr
            if img.shape[1] > config.HISTORY_OVERLAY_MAX_W:
                scale = config.HISTORY_OVERLAY_MAX_W / img.shape[1]
                img = cv2.resize(img, None, fx=scale, fy=scale)
            ts = int(t_wall * 1000)
            fname = f"hist_{ts}_{len(self.entries):05d}.jpg"
            fpath = config.HISTORY_DIR / fname
            cv2.imwrite(str(fpath), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), config.HISTORY_JPEG_QUALITY])
            entry = {
                "file": fname,
                "north": geo_bbox["north"], "south": geo_bbox["south"],
                "east": geo_bbox["east"], "west": geo_bbox["west"],
                "timestamp": meta.get("timestamp", t_wall),
                "utc": meta.get("utc", ""),
                "source": meta.get("source_file", ""),
                "drone_lat": meta.get("drone_lat"), "drone_lon": meta.get("drone_lon"),
                "detections": meta.get("detections", []),
            }
            self.entries.append(entry)
            self._last_save_t = t_wall
            self._prune()
            self._flush()
            return entry
        except Exception as exc:
            logger.error(f"history save failed: {exc}")
            return None

    def _prune(self) -> None:
        while len(self.entries) > config.HISTORY_MAX_FRAMES:
            old = self.entries.pop(0)
            try:
                (config.HISTORY_DIR / old["file"]).unlink(missing_ok=True)
            except Exception:
                pass