"""
src/stitcher.py — Orthorectifying geo-referenced mosaic builder.
================================================================
Offline. Each call to add_frame():

  1. GATES on telemetry — skips frames that don't belong in a macro map
     (AGL too low, or any optical zoom), and frames whose geometry is unreadable.
  2. ORTHORECTIFIES the oblique frame to a north-up nadir tile using a homography
     derived from gimbal tilt/pan/heading + HFOV (the camera↔ground-plane mapping).
  3. PLACES the tile on a geo-referenced master canvas by its ground footprint,
     optionally REFINING the placement with ORB + RANSAC against the existing
     mosaic (corrects residual telemetry error; RANSAC rejects moving cars).
  4. BLENDS with a feathered alpha mask so seams disappear.

GPU path: if the OpenCV build exposes a working cv2.cuda backend, frame upload,
grayscale, ORB feature extraction, descriptor matching and the perspective warp
all run on the RTX 5060 Ti. The stock `opencv-python` pip wheel has NO cuda
backend (getCudaEnabledDeviceCount()==0), so a fully-equivalent CPU path runs by
default. To enable the GPU path you must build OpenCV from source with CUDA for
sm_120 / CUDA 12.8.

Integration contract (see main.py notes):
    tile = stitcher.add_frame(frame, telem)        # north-up ortho tile, or the
                                                   # original frame if gated/bypassed
    bbox = stitcher.last_tile_bounds               # N/S/E/W of that tile (or None)
    stitcher.save_mosaic(path)                     # write the accumulated mosaic
"""
from __future__ import annotations
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

import config

logger = logging.getLogger("Stitcher")

_DEG = math.pi / 180.0
_M_PER_DEG_LAT = 111_111.0


def _cuda_ok() -> bool:
    """True only for an OpenCV built with a working CUDA backend."""
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class OrthoStitcher:
    """Builds a north-up, geo-referenced orthomosaic from oblique UAV frames."""

    def __init__(self, img_w: int = config.FRAME_WIDTH, img_h: int = config.FRAME_HEIGHT,
                 gsd_m: Optional[float] = None):
        self.W = img_w
        self.H = img_h
        self.gsd = gsd_m or config.STITCH_GSD_M

        self.use_gpu = bool(config.STITCH_USE_GPU and _cuda_ok())
        self._gpu_failed = False
        if self.use_gpu:
            try:
                self._orb = cv2.cuda.ORB_create(nfeatures=config.STITCH_ORB_FEATURES)
                self._matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
                logger.info("OrthoStitcher: GPU path active (cv2.cuda ORB + BFMatcher)")
            except Exception as exc:
                logger.warning(f"cv2.cuda init failed ({exc}); using CPU")
                self.use_gpu = False
        if not self.use_gpu:
            self._orb = cv2.ORB_create(nfeatures=config.STITCH_ORB_FEATURES)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            logger.info("OrthoStitcher: CPU path "
                        + ("(pip opencv has no CUDA backend)" if config.STITCH_USE_GPU else "(GPU disabled)"))

        # geo-referenced master canvas (float accumulators for feather blending)
        self._acc: Optional[np.ndarray] = None      # HxWx3 float32 weighted colour sum
        self._wsum: Optional[np.ndarray] = None      # HxW   float32 weight sum
        self._lat0: Optional[float] = None           # origin latitude (for m/deg lon)
        self._north = self._south = self._east = self._west = None
        self.last_tile_bounds: Optional[Dict[str, float]] = None
        self._tiles = 0
        self._skipped = 0

    # ── public API ─────────────────────────────────────────────────────────--
    def add_frame(self, frame: np.ndarray, telem: Dict, geo=None) -> np.ndarray:
        """Orthorectify + stitch one frame. Returns the north-up tile, or the
        original frame unchanged if the frame is gated out / ungeolocatable.

        If a Geolocator `geo` is supplied, its (DEM-corrected) footprint corners
        are used for the rectifying homography, so the ortho tile accounts for
        terrain elevation. Without it, a self-contained flat-earth footprint is
        used."""
        self.last_tile_bounds = None

        if not self._gate(telem):
            self._skipped += 1
            return frame

        corners_en = self._corners_from_geo(telem, geo)
        if corners_en is None:
            try:
                corners_en, _fl, ok = self._ground_corners(telem)
            except Exception as exc:
                logger.debug(f"corner projection failed: {exc}")
                return frame
            if not ok:
                self._skipped += 1
                return frame

        tile, mask, tile_bounds = self._orthorectify(frame, telem, corners_en)
        if tile is None:
            self._skipped += 1
            return frame
        self.last_tile_bounds = tile_bounds

        try:
            self._blend_into_master(tile, mask, tile_bounds)
        except Exception as exc:
            logger.error(f"mosaic blend failed: {exc}")
        self._tiles += 1
        return tile

    def _corners_from_geo(self, telem: Dict, geo) -> Optional[np.ndarray]:
        """DEM-corrected footprint corners (local ENU metres) from the geolocator."""
        if geo is None:
            return None
        try:
            ll = geo.frustum(telem)                # [[lat,lon] x4] TL,TR,BR,BL (DEM-aware)
        except Exception:
            return None
        if not ll or len(ll) != 4:
            return None
        lat0 = float(telem["lat"]); lon0 = float(telem["lon"])
        mlon = _M_PER_DEG_LAT * math.cos(lat0 * _DEG)
        return np.array([[(lon - lon0) * mlon, (lat - lat0) * _M_PER_DEG_LAT]
                         for lat, lon in ll], dtype=np.float64)

    def get_mosaic(self) -> Optional[np.ndarray]:
        if self._acc is None:
            return None
        out = self._acc / np.maximum(self._wsum[..., None], 1e-6)
        return np.clip(out, 0, 255).astype(np.uint8)

    @property
    def mosaic_bounds(self) -> Optional[Dict[str, float]]:
        if self._north is None:
            return None
        return {"north": self._north, "south": self._south,
                "east": self._east, "west": self._west}

    def save_mosaic(self, path: Path = None) -> Optional[Path]:
        path = Path(path or config.MOSAIC_OUTPUT)
        m = self.get_mosaic()
        if m is None:
            logger.warning("no mosaic to save (no frames stitched)")
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), m, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            logger.info(f"orthomosaic saved: {path} ({m.shape[1]}x{m.shape[0]}px) "
                        f"· {self._tiles} tiles · {self._skipped} skipped")
            return path
        except Exception as exc:
            logger.error(f"save_mosaic failed: {exc}")
            return None

    @property
    def stats(self) -> Dict[str, int]:
        return {"tiles": self._tiles, "skipped": self._skipped,
                "gpu": int(self.use_gpu)}

    # ── 1. telemetry gate ─────────────────────────────────────────────────────
    @staticmethod
    def _depression_deg(telem: Dict) -> Optional[float]:
        """Absolute gimbal depression below horizontal (90 = nadir, 0 = horizon).
        Normalises the HUD's negative-down convention and reflects past-nadir."""
        tilt = telem.get("tilt")
        if tilt is None:
            return None
        dep = -float(tilt)                                # HUD prints down-tilt negative
        if dep > 90.0:                                    # past nadir -> reflect
            dep = 180.0 - dep
        return dep

    @classmethod
    def _gate(cls, telem: Dict) -> bool:
        agl = telem.get("agl")
        zoom = telem.get("zoom_o")
        tilt = telem.get("tilt")
        hfov = telem.get("hfov")
        if agl is None or tilt is None or hfov is None:
            return False                                  # need geometry to ortho
        if agl < config.STITCH_MIN_AGL_M:
            return False                                  # too low for a macro map
        if zoom is not None and zoom > config.STITCH_MAX_ZOOM:
            return False                                  # zoomed in: not a wide tile
        dep = cls._depression_deg(telem)
        if dep is None or dep < config.STITCH_MIN_DEPRESSION_DEG:
            return False                                  # too oblique: 2D ortho degenerates
        return True

    # ── 2a. project the 4 image corners to the ground plane (local ENU metres) ─
    def _ground_corners(self, telem: Dict) -> Tuple[np.ndarray, float, bool]:
        agl = float(telem["agl"])
        hfov = max(config.HFOV_MIN_DEG, min(config.HFOV_MAX_DEG,
                                            float(telem.get("hfov") or config.HFOV_FALLBACK_DEG)))
        # normalise the HUD tilt convention (down-tilt printed negative) -> depression
        dep = -float(telem["tilt"])
        az = float(telem.get("heading") or 0.0) + float(telem.get("pan") or 0.0)
        if dep > 90.0:                                    # past nadir: reflect + flip
            dep = 180.0 - dep
            az += 180.0
        if dep < config.MIN_DEPRESSION_DEG:
            return np.empty((0, 2)), 0.0, False

        dr, ar = dep * _DEG, az * _DEG
        b = np.array([math.cos(dr) * math.sin(ar), math.cos(dr) * math.cos(ar), -math.sin(dr)])
        right = _norm(np.cross(np.array([0.0, 0.0, 1.0]), b))
        down = _norm(np.cross(b, right))
        fl = (self.W / 2.0) / math.tan((hfov * _DEG) / 2.0)

        pts = [(0, 0), (self.W, 0), (self.W, self.H), (0, self.H)]   # TL TR BR BL
        en = []
        for px, py in pts:
            u = (px - self.W / 2.0) / fl
            v = (py - self.H / 2.0) / fl
            ray = _norm(b + u * right + v * down)
            if ray[2] >= -1e-4:                           # ray points at/above horizon
                return np.empty((0, 2)), fl, False
            s = agl / (-ray[2])
            if s <= 0 or s > config.MAX_RANGE_M:
                return np.empty((0, 2)), fl, False
            en.append([s * ray[0], s * ray[1]])           # east, north (metres)
        return np.asarray(en, dtype=np.float64), fl, True

    # ── 2b. warp oblique frame -> north-up nadir tile ─────────────────────────
    def _orthorectify(self, frame: np.ndarray, telem: Dict, corners_en: np.ndarray):
        # local-ENU bbox of the footprint -> tile raster at the chosen GSD
        e_min, n_min = corners_en.min(axis=0)
        e_max, n_max = corners_en.max(axis=0)
        tile_w = int(math.ceil((e_max - e_min) / self.gsd))
        tile_h = int(math.ceil((n_max - n_min) / self.gsd))
        if tile_w < 8 or tile_h < 8 or max(tile_w, tile_h) > config.STITCH_MAX_CANVAS_PX:
            return None, None, None

        # destination pixels: north-up (north -> row 0), east -> col increasing
        dst = np.array([[(e - e_min) / self.gsd, (n_max - n) / self.gsd]
                        for e, n in corners_en], dtype=np.float32)
        fh, fw = frame.shape[:2]
        src = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]], dtype=np.float32)
        Hmat = cv2.getPerspectiveTransform(src, dst)

        tile = self._warp(frame, Hmat, (tile_w, tile_h))
        content = np.full(frame.shape[:2], 255, np.uint8)
        if config.STITCH_MASK_HUD:
            fh, fw = frame.shape[:2]
            for (y0, y1, x0, x1) in config.STITCH_HUD_BANDS:
                content[int(y0 * fh):int(y1 * fh), int(x0 * fw):int(x1 * fw)] = 0
        mask = self._warp(content, Hmat, (tile_w, tile_h))

        # footprint centre -> lat/lon, then bbox of the tile in geo coords
        lat0 = float(telem["lat"]); lon0 = float(telem["lon"])
        mlon = _M_PER_DEG_LAT * math.cos(lat0 * _DEG)
        bounds = {
            "north": lat0 + n_max / _M_PER_DEG_LAT,
            "south": lat0 + n_min / _M_PER_DEG_LAT,
            "east":  lon0 + e_max / mlon,
            "west":  lon0 + e_min / mlon,
        }
        return tile, mask, bounds

    # ── GPU/CPU warp ──────────────────────────────────────────────────────────
    def _warp(self, img: np.ndarray, Hmat: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if self.use_gpu and not self._gpu_failed:
            try:
                g = cv2.cuda_GpuMat()
                g.upload(img)
                out = cv2.cuda.warpPerspective(g, Hmat, size, flags=cv2.INTER_LINEAR)
                return out.download()
            except Exception as exc:
                logger.warning(f"cuda warp failed ({exc}); CPU for the rest of the run")
                self._gpu_failed = True
        return cv2.warpPerspective(img, Hmat, size, flags=cv2.INTER_LINEAR)

    # ── feature extraction (GPU or CPU) ───────────────────────────────────────
    def _features(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.use_gpu and not self._gpu_failed:
            try:
                g = cv2.cuda_GpuMat(); g.upload(gray)
                kp_gpu, desc_gpu = self._orb.detectAndComputeAsync(g, None)
                kp = self._orb.convert(kp_gpu)
                desc = desc_gpu.download() if desc_gpu is not None else None
                pts = np.float32([k.pt for k in kp]) if kp else np.empty((0, 2), np.float32)
                return pts, desc
            except Exception as exc:
                logger.warning(f"cuda ORB failed ({exc}); CPU for the rest of the run")
                self._gpu_failed = True
        kp, desc = self._orb.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kp]) if kp else np.empty((0, 2), np.float32)
        return pts, desc

    def _ratio_match(self, descA, descB) -> List:
        if descA is None or descB is None or len(descA) < 2 or len(descB) < 2:
            return []
        try:
            raw = self._matcher.knnMatch(descA, descB, k=2)
        except Exception:
            raw = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descA, descB, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2 and pair[0].distance < config.STITCH_MATCH_RATIO * pair[1].distance:
                good.append(pair[0])
        return good

    # ── 3+4. place tile on the geo master, with optional ORB refinement + blend ─
    def _geo_to_px(self, lat: float, lon: float) -> Tuple[float, float]:
        mlon = _M_PER_DEG_LAT * math.cos(self._lat0 * _DEG)
        col = (lon - self._west) * mlon / self.gsd
        row = (self._north - lat) * _M_PER_DEG_LAT / self.gsd
        return col, row

    def _ensure_canvas(self, b: Dict[str, float]) -> None:
        if self._acc is None:
            self._lat0 = (b["north"] + b["south"]) / 2.0
            self._north, self._south = b["north"], b["south"]
            self._east, self._west = b["east"], b["west"]
            w, h = self._dims()
            self._acc = np.zeros((h, w, 3), np.float32)
            self._wsum = np.zeros((h, w), np.float32)
            return
        new_n = max(self._north, b["north"]); new_s = min(self._south, b["south"])
        new_e = max(self._east, b["east"]);  new_w = min(self._west, b["west"])
        if (new_n, new_s, new_e, new_w) == (self._north, self._south, self._east, self._west):
            return
        old_n, old_w = self._north, self._west
        self._north, self._south, self._east, self._west = new_n, new_s, new_e, new_w
        w, h = self._dims()
        if max(w, h) > config.STITCH_MAX_CANVAS_PX:
            raise RuntimeError(f"mosaic canvas {w}x{h} exceeds cap")
        acc = np.zeros((h, w, 3), np.float32)
        wsum = np.zeros((h, w), np.float32)
        # offset of the old canvas inside the grown one
        off_c = int(round((old_w - self._west) * (_M_PER_DEG_LAT * math.cos(self._lat0 * _DEG)) / self.gsd))
        off_r = int(round((self._north - old_n) * _M_PER_DEG_LAT / self.gsd))
        oh, ow = self._acc.shape[:2]
        acc[off_r:off_r + oh, off_c:off_c + ow] = self._acc
        wsum[off_r:off_r + oh, off_c:off_c + ow] = self._wsum
        self._acc, self._wsum = acc, wsum

    def _dims(self) -> Tuple[int, int]:
        mlon = _M_PER_DEG_LAT * math.cos(self._lat0 * _DEG)
        w = int(math.ceil((self._east - self._west) * mlon / self.gsd))
        h = int(math.ceil((self._north - self._south) * _M_PER_DEG_LAT / self.gsd))
        return max(1, w), max(1, h)

    def _feather(self, mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        fw = float(config.STITCH_FEATHER_PX)
        return np.clip(dist / max(1.0, fw), 0.0, 1.0).astype(np.float32)

    def _blend_into_master(self, tile: np.ndarray, mask: np.ndarray, b: Dict[str, float]) -> None:
        self._ensure_canvas(b)
        col, row = self._geo_to_px(b["north"], b["west"])   # tile top-left -> master px
        c0, r0 = int(round(col)), int(round(row))
        th, tw = tile.shape[:2]
        Hc, Wc = self._acc.shape[:2]
        # clip to canvas
        r1, c1 = max(0, r0), max(0, c0)
        r2, c2 = min(Hc, r0 + th), min(Wc, c0 + tw)
        if r2 <= r1 or c2 <= c1:
            return
        ty1, tx1 = r1 - r0, c1 - c0
        ty2, tx2 = ty1 + (r2 - r1), tx1 + (c2 - c1)
        sub_tile = tile[ty1:ty2, tx1:tx2]
        sub_mask = mask[ty1:ty2, tx1:tx2]

        weight = self._feather(sub_mask)

        # optional ORB + RANSAC refinement against the existing mosaic content
        if config.STITCH_REFINE_WITH_FEATURES and self._tiles > 0:
            try:
                sub_tile, weight = self._refine(sub_tile, sub_mask, weight, r1, c1)
            except Exception as exc:
                logger.debug(f"refine skipped: {exc}")

        self._acc[r1:r2, c1:c2] += sub_tile.astype(np.float32) * weight[..., None]
        self._wsum[r1:r2, c1:c2] += weight

    def _refine(self, tile: np.ndarray, mask: np.ndarray, weight: np.ndarray,
                r1: int, c1: int):
        h, w = tile.shape[:2]
        existing = self.get_mosaic()
        if existing is None:
            return tile, weight
        base = existing[r1:r1 + h, c1:c1 + w]
        if base.shape[:2] != tile.shape[:2] or (base.sum() == 0):
            return tile, weight
        ptsT, descT = self._features(tile)
        ptsB, descB = self._features(base)
        good = self._ratio_match(descT, descB)
        if len(good) < config.STITCH_MIN_MATCHES:
            return tile, weight
        src = np.float32([ptsT[m.queryIdx] for m in good])
        dst = np.float32([ptsB[m.trainIdx] for m in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=config.STITCH_RANSAC_THRESH)
        if M is None or inl is None or int(inl.sum()) < config.STITCH_MIN_MATCHES:
            return tile, weight
        shift_m = math.hypot(M[0, 2], M[1, 2]) * self.gsd
        if shift_m > config.STITCH_REFINE_MAX_SHIFT_M:        # implausible correction -> trust geo
            return tile, weight
        tile2 = cv2.warpAffine(tile, M, (w, h), flags=cv2.INTER_LINEAR)
        mask2 = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        return tile2, self._feather(mask2)