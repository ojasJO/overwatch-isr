"""
src/geolocator.py — Stable 3D ray-rotation geolocation with drift control.
==========================================================================
Turns a detection pixel + HUD telemetry into a ground coordinate (lat/lon +
MGRS) and kills the "ghost movement" / target-jitter drift at its source.

Why the drift happened
-----------------------
The airframe is physically static, yet HUD changes in gimbal tilt/pan/heading
and zoom were being folded into the math in a way that (a) let the *plotted
platform* wander and (b) amplified small angle errors into large ground drift
through the oblique ray-cast. A blind EMA on raw telemetry only smears the
symptom.

The fix, in layers
------------------
1. PLATFORM ISOLATION. The absolute base state (lat, lon, AMSL) is held in a
   stiff low-pass and is the ONLY thing that plots the station marker. Gimbal /
   zoom changes never move it; they only move the aim point (frustum).
2. TILT-CONVENTION AUTO-CALIBRATION. Spec says down-tilt is positive (+10..+90);
   the HUD prints it negative (e.g. -92 near nadir). We bound + normalise that
   inversion, and when the HUD also prints the drone's own TGT we *measure*
   which convention minimises error and lock it — no guessing.
3. 3D RAY ROTATION. The camera ray is rotated into an ENU world frame and
   intersected with the ground plane. Stable at, near and past nadir.
4. TGT-ANCHOR BIAS REMOVAL. The shared systematic boresight error is estimated
   from (our centre estimate − HUD TGT) and subtracted from every detection.
5. TEMPORAL SMOOTHING. A constant-position Kalman filter runs ON THE OUTPUT
   ground coordinate per target, in local ENU metres, weighted by each
   detection's propagated 1-sigma — completely decoupled from raw telemetry.

Telemetry dict keys consumed (any may be missing):
  lat, lon, amsl, agl, heading, tilt, pan, hfov, tgt_lat, tgt_lon, sr
"""
from __future__ import annotations
import math
import logging
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger("Geolocator")

_DEG = math.pi / 180.0
_M_PER_DEG_LAT = 111_111.0


# ════════════════════════════════════════════════════════════════════════════
# MGRS  (self-contained WGS84 lat/lon -> UTM -> MGRS; no external dependency)
# ════════════════════════════════════════════════════════════════════════════
_A = 6_378_137.0                       # WGS84 semi-major axis (m)
_F = 1.0 / 298.257223563               # flattening
_E2 = _F * (2 - _F)                    # eccentricity^2
_E_P2 = _E2 / (1 - _E2)
_K0 = 0.9996

_LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"    # 8° bands from -80; X spans 72..84
_COL_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"  # 20 rows (no I,O), repeats every 2000 km


def _utm_zone(lat: float, lon: float) -> int:
    zone = int((lon + 180.0) / 6.0) + 1
    # Norway / Svalbard exceptions (harmless to include)
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32
    if 72.0 <= lat < 84.0:
        if 0.0 <= lon < 9.0:   zone = 31
        elif 9.0 <= lon < 21.0: zone = 33
        elif 21.0 <= lon < 33.0: zone = 35
        elif 33.0 <= lon < 42.0: zone = 37
    return zone


def _lat_band(lat: float) -> str:
    if lat >= 84.0:
        return "X"
    if lat < -80.0:
        return "C"
    idx = int((lat + 80.0) / 8.0)
    idx = max(0, min(len(_LAT_BANDS) - 1, idx))
    return _LAT_BANDS[idx]


def _latlon_to_utm(lat: float, lon: float) -> Tuple[int, float, float]:
    """Return (zone, easting, northing) in metres (WGS84, Snyder TM)."""
    zone = _utm_zone(lat, lon)
    lon0 = (zone - 1) * 6 - 180 + 3            # central meridian
    lat_r, lon_r, lon0_r = lat * _DEG, lon * _DEG, lon0 * _DEG

    N = _A / math.sqrt(1 - _E2 * math.sin(lat_r) ** 2)
    T = math.tan(lat_r) ** 2
    C = _E_P2 * math.cos(lat_r) ** 2
    A = math.cos(lat_r) * (lon_r - lon0_r)

    M = _A * ((1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256) * lat_r
              - (3 * _E2 / 8 + 3 * _E2 ** 2 / 32 + 45 * _E2 ** 3 / 1024) * math.sin(2 * lat_r)
              + (15 * _E2 ** 2 / 256 + 45 * _E2 ** 3 / 1024) * math.sin(4 * lat_r)
              - (35 * _E2 ** 3 / 3072) * math.sin(6 * lat_r))

    easting = (_K0 * N * (A + (1 - T + C) * A ** 3 / 6
               + (5 - 18 * T + T ** 2 + 72 * C - 58 * _E_P2) * A ** 5 / 120) + 500_000.0)
    northing = (_K0 * (M + N * math.tan(lat_r) * (A ** 2 / 2
                + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                + (61 - 58 * T + T ** 2 + 600 * C - 330 * _E_P2) * A ** 6 / 720)))
    if lat < 0:
        northing += 10_000_000.0
    return zone, easting, northing


def latlon_to_mgrs(lat: float, lon: float, precision: int = 5) -> str:
    """WGS84 (lat, lon) -> MGRS string. precision 5 = 1 m. Returns '' on error."""
    try:
        if not (-80.0 <= lat <= 84.0):
            return ""
        zone, easting, northing = _latlon_to_utm(lat, lon)
        band = _lat_band(lat)

        col_letter = _COL_SETS[(zone - 1) % 3][int(easting // 100_000) - 1]
        row_idx = int((northing % 2_000_000) // 100_000)
        if zone % 2 == 0:                       # even zones shift rows by 5
            row_idx = (row_idx + 5) % 20
        row_letter = _ROW_LETTERS[row_idx % 20]

        precision = max(1, min(5, precision))
        e = int(round(easting % 100_000))
        n = int(round(northing % 100_000))
        div = 10 ** (5 - precision)
        e_str = f"{e // div:0{precision}d}"
        n_str = f"{n // div:0{precision}d}"
        return f"{zone}{band}{col_letter}{row_letter} {e_str} {n_str}"
    except Exception as exc:                     # never crash the pipeline on MGRS
        logger.debug(f"MGRS conversion failed: {exc}")
        return ""


# ════════════════════════════════════════════════════════════════════════════
# vector helpers
# ════════════════════════════════════════════════════════════════════════════
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = lat1 * _DEG, lat2 * _DEG
    dphi = (lat2 - lat1) * _DEG
    dlmb = (lon2 - lon1) * _DEG
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(v):
    m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


# ════════════════════════════════════════════════════════════════════════════
# DEM (Copernicus GLO-30) — terrain elevation lookup for true ground intersection
# ════════════════════════════════════════════════════════════════════════════
_DEM_CACHE: Dict[str, "Optional[_DEM]"] = {}


class _DEM:
    """Loads a GeoTIFF DEM into memory once; bilinearly samples terrain AMSL (m).
    Returns None outside coverage / on nodata so callers can fall back to flat-earth."""

    def __init__(self, path):
        import rasterio
        from rasterio.warp import transform as _rio_transform
        self._rio_transform = _rio_transform
        with rasterio.open(str(path)) as ds:
            self.elev = ds.read(1).astype("float32")
            self.inv = ~ds.transform                       # (x,y) -> (col,row)
            self.crs = ds.crs
            self.nodata = ds.nodata
            self.is_geographic = (ds.crs is None) or bool(ds.crs.is_geographic)
        self.h, self.w = self.elev.shape

    def elevation(self, lat: float, lon: float) -> Optional[float]:
        x, y = lon, lat
        if not self.is_geographic:                         # reproject WGS84 -> DEM CRS
            try:
                xs, ys = self._rio_transform("EPSG:4326", self.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            except Exception:
                return None
        col, row = self.inv * (x, y)
        if col < 0 or row < 0 or col > self.w - 1 or row > self.h - 1:
            return None                                    # outside DEM bounds
        c0, r0 = int(math.floor(col)), int(math.floor(row))
        c1, r1 = min(c0 + 1, self.w - 1), min(r0 + 1, self.h - 1)
        fc, fr = col - c0, row - r0
        v00, v01 = self.elev[r0, c0], self.elev[r0, c1]
        v10, v11 = self.elev[r1, c0], self.elev[r1, c1]
        if self.nodata is not None and self.nodata in (v00, v01, v10, v11):
            return None
        top = v00 * (1 - fc) + v01 * fc
        bot = v10 * (1 - fc) + v11 * fc
        return float(top * (1 - fr) + bot * fr)


def _load_dem(path) -> "Optional[_DEM]":
    key = str(path)
    if key in _DEM_CACHE:
        return _DEM_CACHE[key]
    dem = None
    try:
        from pathlib import Path as _P
        if _P(path).exists():
            dem = _DEM(path)
    except Exception as exc:
        logger.warning(f"DEM load failed ({exc}); flat-earth fallback will be used")
        dem = None
    _DEM_CACHE[key] = dem
    return dem


class _Gimbal:
    """Lightweight telemetry stand-in used for projection + perturbation."""
    __slots__ = ("lat", "lon", "agl", "heading", "tilt", "pan", "hfov", "amsl")

    def __init__(self, lat, lon, agl, heading, tilt, pan, hfov, amsl=None):
        self.lat, self.lon, self.agl = lat, lon, agl
        self.heading, self.tilt, self.pan, self.hfov = heading, tilt, pan, hfov
        self.amsl = amsl


class _Track:
    """Constant-position Kalman in local ENU metres (per detected target)."""
    __slots__ = ("id", "lat0", "lon0", "n", "e", "cls", "sigma", "misses")

    def __init__(self, tid: int, lat: float, lon: float, cls: str, sigma: float):
        self.id, self.lat0, self.lon0, self.cls = tid, lat, lon, cls
        self.n = self.e = 0.0
        self.sigma = sigma
        self.misses = 0

    def latlon(self) -> Tuple[float, float]:
        lat = self.lat0 + self.n / _M_PER_DEG_LAT
        lon = self.lon0 + self.e / (_M_PER_DEG_LAT * math.cos(self.lat0 * _DEG))
        return lat, lon

    def update(self, lat: float, lon: float, r_m: float) -> None:
        zn = (lat - self.lat0) * _M_PER_DEG_LAT
        ze = (lon - self.lon0) * _M_PER_DEG_LAT * math.cos(self.lat0 * _DEG)
        q = config.TRACK_Q_M ** 2
        r = max(config.TRACK_MIN_R_M, r_m) ** 2
        p = self.sigma ** 2 + q
        k = p / (p + r)
        self.n += k * (zn - self.n)
        self.e += k * (ze - self.e)
        self.sigma = math.sqrt(max(1e-6, (1 - k) * p))
        self.misses = 0


# ════════════════════════════════════════════════════════════════════════════
# Geolocator
# ════════════════════════════════════════════════════════════════════════════
class Geolocator:
    def __init__(self, img_w: int = config.FRAME_WIDTH, img_h: int = config.FRAME_HEIGHT):
        self.W = img_w
        self.H = img_h
        # platform isolation state
        self._plat_lat: Optional[float] = None
        self._plat_lon: Optional[float] = None
        self._plat_amsl: Optional[float] = None
        # tilt convention calibration
        self._conv = "neg_nadir"
        self._conv_locked = False
        self._conv_err = {"neg_nadir": 0.0, "pos_down": 0.0}
        self._conv_n = 0
        # bias state
        self._bias_n = 0.0
        self._bias_e = 0.0
        self._bias_valid = False
        # tracker
        self._tracks: List[_Track] = []
        self._next_id = 1
        # terrain (DEM) — terrain-aware ray-cast with graceful flat-earth fallback
        self._dem = _load_dem(config.DEM_PATH) if config.USE_DEM else None
        if self._dem is not None:
            logger.info(f"DEM active: {config.DEM_PATH.name} "
                        f"({self._dem.w}x{self._dem.h}) — terrain ray-cast ON")
        else:
            logger.info("DEM not loaded — flat-earth AGL ray-cast (fallback mode)")
        logger.info(f"Geolocator ready · {self.W}x{self.H} · ray-rotation · "
                    f"alt_ref={config.ALTITUDE_REF} · TGT_anchor={config.USE_TGT_ANCHOR}")

    # ── platform isolation ────────────────────────────────────────────────────
    def update_platform(self, telem: Dict) -> Optional[Tuple[float, float, float]]:
        """Stiff low-pass on the static airframe position. Returns (lat, lon, amsl)."""
        lat, lon = telem.get("lat"), telem.get("lon")
        amsl = telem.get("amsl")
        if lat is None or lon is None:
            if self._plat_lat is None:
                return None
            return self._plat_lat, self._plat_lon, self._plat_amsl
        a = config.PLATFORM_EMA_ALPHA
        if self._plat_lat is None:
            self._plat_lat, self._plat_lon = lat, lon
            self._plat_amsl = amsl if amsl is not None else 0.0
        else:
            self._plat_lat += a * (lat - self._plat_lat)
            self._plat_lon += a * (lon - self._plat_lon)
            if amsl is not None:
                self._plat_amsl += a * (amsl - self._plat_amsl)
        return self._plat_lat, self._plat_lon, self._plat_amsl

    def platform_point(self) -> Optional[Tuple[float, float]]:
        if self._plat_lat is None:
            return None
        return round(self._plat_lat, 7), round(self._plat_lon, 7)

    # ── intrinsics ─────────────────────────────────────────────────────────--
    def _focal_px(self, hfov_deg: float) -> float:
        hfov_deg = max(config.HFOV_MIN_DEG, min(config.HFOV_MAX_DEG, hfov_deg))
        return (self.W / 2.0) / math.tan((hfov_deg * _DEG) / 2.0)

    # ── geolocatability ────────────────────────────────────────────────────--
    @staticmethod
    def _geolocatable(telem: Dict) -> bool:
        return (telem.get("lat") is not None and telem.get("lon") is not None
                and telem.get("agl") is not None and float(telem.get("agl") or 0) > 0
                and telem.get("tilt") is not None)

    def _build_gimbal(self, telem: Dict) -> _Gimbal:
        # use the ISOLATED (smoothed) platform position so gimbal jitter cannot
        # translate into platform/target wander
        lat = self._plat_lat if self._plat_lat is not None else telem["lat"]
        lon = self._plat_lon if self._plat_lon is not None else telem["lon"]
        amsl = self._plat_amsl if self._plat_amsl is not None else telem.get("amsl")
        hfov = telem.get("hfov") or config.HFOV_FALLBACK_DEG
        return _Gimbal(lat, lon, float(telem["agl"]), float(telem.get("heading") or 0.0),
                       float(telem["tilt"]), float(telem.get("pan") or 0.0), float(hfov),
                       amsl=(float(amsl) if amsl is not None else None))

    # ── core projection: pixel -> (lat, lon, horiz_m) or None ─────────────────
    def _project(self, px: float, py: float, g: _Gimbal, conv: Optional[str] = None):
        conv = conv or self._conv
        fl = self._focal_px(g.hfov)

        raw_dep = (-g.tilt) if conv == "neg_nadir" else (g.tilt)   # normalise inversion
        az0 = g.heading + g.pan
        if raw_dep > 90.0:                       # past nadir -> reflect and flip azimuth
            raw_dep = 180.0 - raw_dep
            az0 += 180.0
        if raw_dep < config.MIN_DEPRESSION_DEG:
            return None

        dep = raw_dep * _DEG
        az = az0 * _DEG
        b = (math.cos(dep) * math.sin(az), math.cos(dep) * math.cos(az), -math.sin(dep))
        right = _norm(_cross((0.0, 0.0, 1.0), b))
        down = _norm(_cross(b, right))

        u = (px - self.W / 2.0) / fl
        v = (py - self.H / 2.0) / fl
        ray = _norm((b[0] + u * right[0] + v * down[0],
                     b[1] + u * right[1] + v * down[1],
                     b[2] + u * right[2] + v * down[2]))
        if ray[2] >= -1e-4:
            return None

        # ── terrain-aware intersection (DEM) with flat-earth fallback ──────────
        if self._dem is not None and g.amsl is not None:
            hit = self._intersect_dem(g.amsl, ray, g.lat, g.lon)
            if hit is not None:
                return hit

        # ── flat-earth intersection: ground plane at AGL below the platform ────
        s = g.agl / (-ray[2])
        if s <= 0 or s > config.MAX_RANGE_M:
            return None
        east = s * ray[0]
        north = s * ray[1]
        lat = g.lat + north / _M_PER_DEG_LAT
        lon = g.lon + east / (_M_PER_DEG_LAT * math.cos(g.lat * _DEG))
        return lat, lon, math.hypot(east, north)

    # ── DEM ray-march: first crossing of the camera ray below the terrain ──────
    def _elev(self, lat: float, lon: float) -> Optional[float]:
        if self._dem is None:
            return None
        try:
            return self._dem.elevation(lat, lon)
        except Exception:
            return None

    def _intersect_dem(self, origin_amsl: float, ray, lat0: float, lon0: float):
        """March the ray (east, north, up) from the platform until it drops to/below
        the DEM surface; bisection-refine the crossing. Returns (lat, lon, horiz_m)
        or None to signal a fallback (no DEM hit, or ray left DEM coverage)."""
        e, n, up = ray
        if up >= -1e-6:
            return None
        coslat = math.cos(lat0 * _DEG)

        def at(t: float):
            E, N = t * e, t * n
            lat = lat0 + N / _M_PER_DEG_LAT
            lon = lon0 + E / (_M_PER_DEG_LAT * coslat)
            return lat, lon, E, N

        lat_, lon_, _, _ = at(1e-3)
        g0 = self._elev(lat_, lon_)
        if g0 is None or (origin_amsl - g0) <= 0:          # no coverage / already below
            return None

        step = max(0.5, float(config.DEM_MARCH_STEP_M))
        prev_t = 0.0
        t = step
        while t <= config.MAX_RANGE_M:
            lat_, lon_, E, N = at(t)
            gt = self._elev(lat_, lon_)
            if gt is None:                                 # left DEM coverage -> fallback
                return None
            if (origin_amsl + t * up) - gt <= 0.0:         # crossed below terrain
                a, b = prev_t, t
                for _ in range(int(config.DEM_BISECT_ITERS)):
                    m = 0.5 * (a + b)
                    lm, om, _, _ = at(m)
                    gm = self._elev(lm, om)
                    if gm is None:
                        break
                    if (origin_amsl + m * up) - gm > 0.0:
                        a = m
                    else:
                        b = m
                tm = 0.5 * (a + b)
                lm, om, Em, Nm = at(tm)
                return lm, om, math.hypot(Em, Nm)
            prev_t = t
            t += step
        return None

    # ── 1-sigma ground error via finite-difference propagation ────────────────
    def _sigma_m(self, px: float, py: float, g: _Gimbal) -> float:
        base = self._project(px, py, g)
        if base is None:
            return float("inf")
        bl, bo, _ = base

        def shift(attr, d):
            kw = {k: getattr(g, k) for k in ("lat", "lon", "agl", "heading", "tilt", "pan", "hfov", "amsl")}
            kw[attr] += d
            return _Gimbal(**kw)

        var = config.SIGMA_GNSS_M ** 2
        for attr, sig in (("tilt", config.SIGMA_TILT_DEG), ("pan", config.SIGMA_PAN_DEG),
                          ("heading", config.SIGMA_HDG_DEG)):
            p = self._project(px, py, shift(attr, sig))
            if p:
                var += haversine_m(bl, bo, p[0], p[1]) ** 2
        p = self._project(px, py, shift("agl", g.agl * config.SIGMA_AGL_FRAC))
        if p:
            var += haversine_m(bl, bo, p[0], p[1]) ** 2
        return math.sqrt(var)

    # ── tilt-convention auto-calibration vs HUD TGT ───────────────────────────
    def _calibrate(self, telem: Dict, g: _Gimbal) -> None:
        if self._conv_locked or telem.get("tgt_lat") is None or telem.get("tgt_lon") is None:
            return
        cx, cy = config.VALIDATION_PIXEL
        for conv in ("neg_nadir", "pos_down"):
            p = self._project(cx, cy, g, conv=conv)
            if p:
                self._conv_err[conv] += haversine_m(p[0], p[1], telem["tgt_lat"], telem["tgt_lon"])
        self._conv_n += 1
        if self._conv_n >= 10:
            self._conv = min(self._conv_err, key=self._conv_err.get)
            self._conv_locked = True
            logger.info(f"tilt convention locked -> {self._conv} (errs {self._conv_err})")

    # ── TGT-anchor bias estimate ──────────────────────────────────────────────
    def _update_bias(self, telem: Dict, g: _Gimbal) -> None:
        if not config.USE_TGT_ANCHOR or telem.get("tgt_lat") is None or telem.get("tgt_lon") is None:
            return
        cx, cy = config.VALIDATION_PIXEL
        p = self._project(cx, cy, g)
        if p is None:
            return
        dn = (telem["tgt_lat"] - p[0]) * _M_PER_DEG_LAT
        de = (telem["tgt_lon"] - p[1]) * _M_PER_DEG_LAT * math.cos(g.lat * _DEG)
        if math.hypot(dn, de) > config.BIAS_MAX_M:
            return
        a = config.BIAS_LP_ALPHA
        self._bias_n = (1 - a) * self._bias_n + a * dn if self._bias_valid else dn
        self._bias_e = (1 - a) * self._bias_e + a * de if self._bias_valid else de
        self._bias_valid = True

    def _apply_bias(self, lat: float, lon: float, ref_lat: float) -> Tuple[float, float]:
        if not (config.USE_TGT_ANCHOR and self._bias_valid):
            return lat, lon
        lat2 = lat + self._bias_n / _M_PER_DEG_LAT
        lon2 = lon + self._bias_e / (_M_PER_DEG_LAT * math.cos(ref_lat * _DEG))
        return lat2, lon2

    # ── tracker ────────────────────────────────────────────────────────────--
    def _track(self, lat: float, lon: float, cls: str, sigma: float):
        if not config.TRACK_ENABLE:
            return lat, lon, None, sigma
        best, best_d = None, config.TRACK_ASSOC_M
        for tr in self._tracks:
            if tr.cls != cls:
                continue
            tl, to = tr.latlon()
            d = haversine_m(lat, lon, tl, to)
            if d < best_d:
                best, best_d = tr, d
        if best is None:
            best = _Track(self._next_id, lat, lon, cls, sigma)
            self._next_id += 1
            self._tracks.append(best)
        best.update(lat, lon, sigma)
        sl, so = best.latlon()
        return sl, so, best.id, best.sigma

    def _age_tracks(self, seen: set) -> None:
        for tr in self._tracks:
            if tr.id not in seen:
                tr.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= config.TRACK_MAX_AGE_FRAMES]

    # ── public: geolocate a frame of detections ───────────────────────────────
    def locate(self, detections: List[Dict], telem: Dict) -> List[Dict]:
        if not self._geolocatable(telem):
            for d in detections:
                d.update(lat=None, lon=None, mgrs="", accuracy_m=None, track_id=None)
            return detections

        g = self._build_gimbal(telem)
        self._calibrate(telem, g)
        self._update_bias(telem, g)

        seen = set()
        for d in detections:
            p = self._project(d["center_x"], d["center_y"], g)
            if p is None:
                d.update(lat=None, lon=None, mgrs="", accuracy_m=None, track_id=None)
                continue
            lat, lon = self._apply_bias(p[0], p[1], g.lat)
            sigma = self._sigma_m(d["center_x"], d["center_y"], g)
            lat, lon, tid, sigma = self._track(lat, lon, d.get("class_name", "?"), sigma)
            d["lat"] = round(lat, 7)
            d["lon"] = round(lon, 7)
            d["mgrs"] = latlon_to_mgrs(lat, lon, config.MGRS_PRECISION)
            d["accuracy_m"] = round(sigma, 1)
            d["track_id"] = tid
            if tid is not None:
                seen.add(tid)
        self._age_tracks(seen)
        return detections

    # ── viewing frustum (cone polygon on the ground) ──────────────────────────
    def frustum(self, telem: Dict) -> List[List[float]]:
        """Four ground corners [[lat,lon],...] of the camera footprint, in order
        TL,TR,BR,BL. Empty list if geometry is unreliable."""
        if not self._geolocatable(telem):
            return []
        g = self._build_gimbal(telem)
        order = [(0, 0), (self.W, 0), (self.W, self.H), (0, self.H)]
        poly: List[List[float]] = []
        for px, py in order:
            p = self._project(px, py, g)
            if p is None:
                return []
            poly.append([round(p[0], 7), round(p[1], 7)])
        return poly

    def geo_bbox(self, telem: Dict) -> Optional[Dict[str, float]]:
        """North/South/East/West bounds of the current footprint (for history)."""
        poly = self.frustum(telem)
        if not poly:
            return None
        lats = [c[0] for c in poly]
        lons = [c[1] for c in poly]
        return {"north": max(lats), "south": min(lats), "east": max(lons), "west": min(lons)}

    # ── boresight uncertainty (the dashboard's "error margin") ─────────────────
    def boresight_error_m(self, telem: Dict) -> Optional[float]:
        if not self._geolocatable(telem):
            return None
        g = self._build_gimbal(telem)
        cx, cy = config.VALIDATION_PIXEL
        s = self._sigma_m(cx, cy, g)
        return None if s == float("inf") else round(s, 1)

    def depression_deg(self, telem: Dict) -> Optional[float]:
        """Normalised gimbal depression below horizontal (for the telemetry grid)."""
        if telem.get("tilt") is None:
            return None
        raw = (-float(telem["tilt"])) if self._conv == "neg_nadir" else float(telem["tilt"])
        if raw > 90.0:
            raw = 180.0 - raw
        return round(raw, 1)

    # ── validation vs HUD TGT (honest: raw centre projection, pre-bias) ────────
    def validate(self, telem: Dict) -> Dict:
        if telem.get("tgt_lat") is None or not self._geolocatable(telem):
            return {}
        g = self._build_gimbal(telem)
        cx, cy = config.VALIDATION_PIXEL
        p = self._project(cx, cy, g)
        if p is None:
            return {}
        err = haversine_m(p[0], p[1], telem["tgt_lat"], telem["tgt_lon"])
        return {
            "error_m": round(err, 1),
            "pass": err < config.VALIDATION_THRESHOLD_M,
            "threshold_m": config.VALIDATION_THRESHOLD_M,
            "bias_m": round(math.hypot(self._bias_n, self._bias_e), 1) if self._bias_valid else None,
            "tilt_convention": self._conv,
        }


# ════════════════════════════════════════════════════════════════════════════
# Master Track Registry — persistent same-class centroid clustering
# ════════════════════════════════════════════════════════════════════════════
class MasterTrackRegistry:
    """Collapses repeated detections of one physical object — seen from many
    gimbal angles across the whole sortie — into a single averaged 'Master Track'.

    A simple, persistent, single-link distance clusterer: a detection is merged
    into the nearest existing track of the SAME class within `merge_m` metres
    (default 15 m), updating that track's running-mean centroid; otherwise it
    seeds a new track. The result is ONE pin per object, not a scattered cluster
    of parallax-drifted dots. Tracks persist for the whole run (no age-out), so
    the map shows everything observed."""

    def __init__(self, merge_m: Optional[float] = None):
        self.merge_m = float(merge_m if merge_m is not None
                             else getattr(config, "MASTER_MERGE_M", 15.0))
        self._tracks: List[Dict] = []
        self._next = 1

    def observe(self, detections: List[Dict], t_now: float) -> None:
        for d in detections:
            lat, lon = d.get("lat"), d.get("lon")
            if lat is None or lon is None:
                continue
            cls = d.get("class_name", "?")
            conf = float(d.get("confidence") or 0.0)
            best, best_d = None, self.merge_m
            for tr in self._tracks:
                if tr["class_name"] != cls:
                    continue
                dist = haversine_m(lat, lon, tr["lat"], tr["lon"])
                if dist < best_d:
                    best, best_d = tr, dist
            if best is None:
                self._tracks.append({
                    "id": self._next, "class_name": cls,
                    "lat": float(lat), "lon": float(lon), "n": 1,
                    "conf_max": conf, "conf_sum": conf,
                    "first": t_now, "last": t_now,
                    "mgrs": latlon_to_mgrs(lat, lon, getattr(config, "MGRS_PRECISION", 5)),
                })
                self._next += 1
            else:
                n = best["n"] + 1
                best["lat"] = (best["lat"] * best["n"] + float(lat)) / n   # running centroid
                best["lon"] = (best["lon"] * best["n"] + float(lon)) / n
                best["n"] = n
                best["conf_max"] = max(best["conf_max"], conf)
                best["conf_sum"] += conf
                best["last"] = t_now
                best["mgrs"] = latlon_to_mgrs(best["lat"], best["lon"],
                                              getattr(config, "MGRS_PRECISION", 5))

    def tracks(self) -> List[Dict]:
        """One de-duplicated, averaged Master Track per physical object."""
        out = []
        for t in self._tracks:
            out.append({
                "track_id": t["id"], "class_name": t["class_name"],
                "lat": round(t["lat"], 7), "lon": round(t["lon"], 7),
                "mgrs": t["mgrs"], "count": t["n"],
                "confidence": round(t["conf_max"], 3),
                "first_seen": t["first"], "last_seen": t["last"],
            })
        return out

    @property
    def n_tracks(self) -> int:
        return len(self._tracks)