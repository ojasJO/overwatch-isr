"""
src/detector.py — YOLOv11 local inference (TensorRT-ready).
===========================================================
Runs on the RTX 5060 Ti (Blackwell sm_120, CUDA 12.8). Loads a TensorRT
`.engine` for maximum throughput when present, otherwise a `.pt` checkpoint
(auto-downloading the stock model as a last resort). Optional SAHI-style
sliced inference recovers tiny top-down objects when zoomed out.

torch / ultralytics are imported lazily inside __init__ so the rest of the
app (and the dashboard) can be imported without a CUDA stack present.
"""
from __future__ import annotations
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import config

logger = logging.getLogger("Detector")


class Detector:
    def __init__(self):
        try:
            import torch
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Detector requires torch + ultralytics. Install with:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
                "  pip install ultralytics\n"
                f"(import error: {exc})"
            )
        self._torch = torch

        engine = Path(config.MODEL_ENGINE)
        pt = Path(config.MODEL_PT)
        if engine.exists():
            model_file = str(engine)
            logger.info(f"Loading TensorRT engine: {model_file}")
        elif pt.exists():
            model_file = str(pt)
            logger.info(f"Loading checkpoint: {model_file}")
        else:
            model_file = f"yolo11{config.MODEL_FALLBACK_SIZE}.pt"
            logger.warning(f"No local weights found — using stock {model_file} (will download)")

        self.model = YOLO(model_file)
        self.is_engine = str(model_file).endswith(".engine")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            p = torch.cuda.get_device_properties(0)
            cap = torch.cuda.get_device_capability(0)
            logger.info(f"GPU {p.name} {p.total_memory/1e9:.1f}GB sm_{cap[0]}{cap[1]} "
                        f"({'Blackwell' if cap[0] >= 12 else 'pre-Blackwell'})")
        else:
            logger.warning("CUDA not available — running on CPU (slow)")

        self._inf_ms: List[float] = []
        try:
            self._warmup()
        except Exception as exc:
            logger.debug(f"warmup skipped: {exc}")
        logger.info("Detector ready")

    def _warmup(self) -> None:
        dummy = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), np.uint8)
        for _ in range(3):
            self._infer(dummy)

    # ── single forward pass ───────────────────────────────────────────────────
    def _infer(self, img: np.ndarray):
        kw = dict(conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD,
                  classes=config.TARGET_CLASSES, verbose=False, imgsz=config.IMG_SIZE)
        if not self.is_engine:                  # engines bake in device/precision
            kw["device"] = self.device
            kw["half"] = (config.USE_HALF and self.device == "cuda")
        return self.model(img, **kw)

    def _boxes(self, results, ox: int = 0, oy: int = 0) -> List[Dict]:
        out: List[Dict] = []
        if results and results[0].boxes is not None:
            names = results[0].names if hasattr(results[0], "names") else self.model.names
            for b in results[0].boxes:
                cid = int(b.cls[0].item())
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(float)
                x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
                raw_name = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
                out.append({
                    "class_id": cid,
                    "class_name": config.CLASS_NAMES.get(cid, raw_name.upper()),
                    "confidence": round(float(b.conf[0].item()), 3),
                    "bbox": [x1, y1, x2, y2],
                    "center_x": (x1 + x2) / 2.0,
                    "center_y": (y1 + y2) / 2.0,
                })
        return out

    @staticmethod
    def _nms(dets: List[Dict], iou_thr: float) -> List[Dict]:
        if not dets:
            return dets
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        keep: List[Dict] = []

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / ua if ua > 0 else 0.0

        for d in dets:
            if all(d["class_id"] != k["class_id"] or iou(d, k) < iou_thr for k in keep):
                keep.append(d)
        return keep

    def _sliced(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        tile, ov = config.SAHI_TILE, config.SAHI_OVERLAP
        step = max(1, int(tile * (1 - ov)))
        dets: List[Dict] = []
        for y in range(0, max(1, h - 1), step):
            for x in range(0, max(1, w - 1), step):
                crop = frame[y:min(y + tile, h), x:min(x + tile, w)]
                if crop.size:
                    dets += self._boxes(self._infer(crop), ox=x, oy=y)
        dets += self._boxes(self._infer(frame))
        return self._nms(dets, config.IOU_THRESHOLD)

    # ── public ────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray, hfov: Optional[float] = None) -> List[Dict]:
        t0 = time.perf_counter()
        try:
            use_tiles = (config.USE_SAHI and hfov is not None and hfov >= config.SAHI_MIN_HFOV)
            dets = self._sliced(frame) if use_tiles else self._boxes(self._infer(frame))
        except Exception as exc:
            logger.error(f"inference failed: {exc}")
            dets = []
        ms = (time.perf_counter() - t0) * 1000.0
        self._inf_ms.append(ms)
        if len(self._inf_ms) > 60:
            self._inf_ms.pop(0)
        for d in dets:
            d["inf_ms"] = round(ms, 1)
        return dets

    @property
    def avg_inf_ms(self) -> float:
        return sum(self._inf_ms) / max(1, len(self._inf_ms))

    def gpu_mem_mb(self) -> float:
        try:
            if self._torch.cuda.is_available():
                return round(self._torch.cuda.memory_reserved() / 1e6, 1)
        except Exception:
            pass
        return 0.0

    # ── annotation for the live viewport ──────────────────────────────────────
    def annotate(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            try:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            except Exception:
                continue
            color = config.CLASS_COLORS.get(det.get("class_name", ""), config.DEFAULT_COLOR)
            acc = f" +/-{det['accuracy_m']}m" if det.get("accuracy_m") is not None else ""
            label = f"{det.get('class_name','?')} {det.get('confidence',0):.0%}{acc}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ytxt = max(y1 - th - 6, 0)
            cv2.rectangle(out, (x1, ytxt), (x1 + tw + 8, ytxt + th + 8), color, -1)
            cv2.putText(out, label, (x1 + 4, ytxt + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (8, 12, 18), 1, cv2.LINE_AA)
        return out