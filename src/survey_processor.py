#!/usr/bin/env python3
"""
src/survey_processor.py — Post-flight Structure-from-Motion (SfM) trigger.
=========================================================================
Real-time SfM is infeasible on the edge node, so this is a STANDALONE,
post-sortie batch tool. It reads history/manifest.json, selects the
highest-quality geo-tagged frames, stages them into an OpenDroneMap (ODM /
WebODM) project, writes an ODM `geo.txt` for georeferencing, and emits a
CUDA-enabled Docker command that uses ODM Split-Merge to cap memory on the
16 GB host.

Run (prepare only — recommended first):
    python -m src.survey_processor
    python src/survey_processor.py --source history --out survey --chunk-size 300

Then launch the generated script:           bash survey/run_survey.sh
or let this tool launch Docker directly:     python src/survey_processor.py --run

NOTES (read once):
  * ODM is a true SfM/MVS pipeline — it expects ORIGINAL perspective frames and
    runs its own bundle adjustment. The /history tiles are ortho-rectified and
    HUD-masked; feeding them in still works but is sub-optimal (double
    rectification). For best survey results, point --source at a directory of
    RAW frames if you have them. The tool accepts either.
  * GPU: ODM uses CUDA principally for SIFT feature extraction (and OpenMVS
    steps where the build supports it). It is NOT end-to-end GPU. The command
    uses the `opendronemap/odm:gpu` image with `--gpus all`, which requires the
    NVIDIA Container Toolkit on the host.
  * OOM control: with > chunk-size images ODM Split-Merge (`--split N
    --split-overlap`) partitions the reconstruction into ~N-image submodels and
    merges them, bounding peak RAM — the supported way to "chunk" large sets.
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

import config

try:
    import piexif
    _HAVE_PIEXIF = True
except Exception:
    _HAVE_PIEXIF = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ survey │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("survey")


# survey-specific knobs read from config if present, else sensible defaults
def _cfg(name: str, default):
    return getattr(config, name, default)


DEFAULTS = {
    "chunk_size":     _cfg("SURVEY_CHUNK_SIZE", 300),       # ODM submodel size
    "split_overlap":  _cfg("SURVEY_SPLIT_OVERLAP_M", 75.0), # metres of submodel overlap
    "cell_m":         _cfg("SURVEY_DEDUPE_CELL_M", 8.0),    # spatial dedupe grid (m)
    "keep_per_cell":  _cfg("SURVEY_KEEP_PER_CELL", 2),      # sharpest N kept per cell
    "max_images":     _cfg("SURVEY_MAX_IMAGES", 3000),      # hard cap
    "default_alt_m":  _cfg("SURVEY_DEFAULT_ALT_M", 0.0),    # fallback geo.txt altitude
    "min_depression": _cfg("STITCH_MIN_DEPRESSION_DEG", 45.0),  # drop oblique frames
    "min_agl":        _cfg("STITCH_MIN_AGL_M", 50.0),          # drop low-altitude frames
    "min_sharpness":  _cfg("SURVEY_MIN_SHARPNESS", 0.0),       # absolute blur floor (0 = rank-only)
    "min_confidence": _cfg("SURVEY_MIN_CONFIDENCE", 0.0),      # drop frames whose best detection < this (0 = off)
    "odm_image":      _cfg("SURVEY_ODM_IMAGE", "opendronemap/odm:gpu"),
    "ortho_res_cm":   _cfg("SURVEY_ORTHO_RES_CM", 5.0),     # cm/px output
    "pc_quality":     _cfg("SURVEY_PC_QUALITY", "medium"),  # OOM-friendly on 16 GB
    "feature_quality":_cfg("SURVEY_FEATURE_QUALITY", "high"),
}
_M_PER_DEG_LAT = 111_111.0


# ── manifest ──────────────────────────────────────────────────────────────────
def load_manifest(source_dir: Path) -> List[Dict]:
    manifest = source_dir / "manifest.json"
    if not manifest.exists():
        log.error(f"no manifest at {manifest}")
        return []
    try:
        data = json.loads(manifest.read_text())
        log.info(f"manifest: {len(data)} entries in {source_dir}")
        return data
    except Exception as exc:
        log.error(f"failed to parse manifest: {exc}")
        return []


def _is_geotagged(e: Dict) -> bool:
    for k in ("north", "south", "east", "west"):
        v = e.get(k)
        if v is None or not math.isfinite(float(v)):
            return False
    return True


def _camera_lonlat(e: Dict) -> Tuple[float, float]:
    """Camera position for geo.txt: drone position if present, else footprint centre."""
    if e.get("drone_lat") is not None and e.get("drone_lon") is not None:
        return float(e["drone_lon"]), float(e["drone_lat"])
    return ((float(e["east"]) + float(e["west"])) / 2.0,
            (float(e["north"]) + float(e["south"])) / 2.0)


def _passes_geometry(e: Dict, min_dep: float, min_agl: float) -> bool:
    """Reject oblique / low-altitude frames. History tiles are already stitcher-gated
    (depression >= 45, AGL >= 50); this re-applies the rule for any manifest that also
    records per-frame `tilt` / `agl` (or raw-frame exports), and is a no-op otherwise."""
    agl = e.get("agl")
    if agl is not None and float(agl) < min_agl:
        return False
    tilt = e.get("tilt")
    if tilt is not None:
        dep = -float(tilt)
        if dep > 90.0:
            dep = 180.0 - dep
        if dep < min_dep:
            return False
    return True


def _best_confidence(e: Dict) -> Optional[float]:
    dets = e.get("detections") or []
    confs = [float(d["confidence"]) for d in dets if d.get("confidence") is not None]
    return max(confs) if confs else None


def _deg_to_dms_rational(value: float):
    """Decimal degrees -> EXIF ((deg,1),(min,1),(sec*100,100)) rationals (abs value)."""
    v = abs(value)
    d = int(v)
    m = int((v - d) * 60)
    s = round((v - d - m / 60.0) * 3600.0, 2)
    return ((d, 1), (m, 1), (int(s * 100), 100))


def write_exif_gps(jpg_path: Path, lat: float, lon: float, alt: float) -> bool:
    """Write GPS lat/lon/alt EXIF into the JPEG via piexif. Returns True on success."""
    if not _HAVE_PIEXIF:
        return False
    try:
        gps = {
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
            piexif.GPSIFD.GPSLatitude: _deg_to_dms_rational(lat),
            piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
            piexif.GPSIFD.GPSLongitude: _deg_to_dms_rational(lon),
            piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1,
            piexif.GPSIFD.GPSAltitude: (int(round(abs(alt) * 100)), 100),
        }
        exif_bytes = piexif.dump({"GPS": gps})
        piexif.insert(exif_bytes, str(jpg_path))
        return True
    except Exception as exc:
        log.debug(f"EXIF write failed for {jpg_path.name}: {exc}")
        return False


def _sharpness(path: Path) -> float:
    """Variance of the Laplacian — higher = sharper. 0 on read failure."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        if img.shape[1] > 640:
            s = 640.0 / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s)
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


# ── selection: geo filter + sharpness rank + spatial dedupe ────────────────────
def select_frames(entries: List[Dict], source_dir: Path,
                  cell_m: float, keep_per_cell: int, max_images: int,
                  min_dep: float = 45.0, min_agl: float = 50.0,
                  min_sharp: float = 0.0, min_conf: float = 0.0) -> List[Dict]:
    cand: List[Dict] = []
    skipped_geom = skipped_conf = skipped_blur = 0
    for e in entries:
        if not _is_geotagged(e):
            continue
        if not _passes_geometry(e, min_dep, min_agl):
            skipped_geom += 1
            continue
        if min_conf > 0.0:
            bc = _best_confidence(e)
            if bc is not None and bc < min_conf:    # drop only frames that HAVE weak detections
                skipped_conf += 1
                continue
        f = source_dir / e["file"]
        if not f.exists():
            continue
        sharp = _sharpness(f)
        if min_sharp > 0.0 and sharp < min_sharp:
            skipped_blur += 1
            continue
        e = dict(e)
        e["_sharp"] = sharp
        e["_lon"], e["_lat"] = _camera_lonlat(e)
        cand.append(e)
    log.info(f"{len(cand)} candidate frames "
             f"(dropped: {skipped_geom} oblique/low-AGL, {skipped_blur} blurry, {skipped_conf} low-confidence)")
    if not cand:
        return []

    # spatial dedupe: bin by footprint centre, keep the sharpest few per cell
    lat0 = sum(c["_lat"] for c in cand) / len(cand)
    mlon = _M_PER_DEG_LAT * math.cos(lat0 * math.pi / 180.0)
    deg_lat = cell_m / _M_PER_DEG_LAT
    deg_lon = cell_m / mlon
    bins: Dict[Tuple[int, int], List[Dict]] = {}
    for c in cand:
        key = (int(c["_lat"] / deg_lat), int(c["_lon"] / deg_lon))
        bins.setdefault(key, []).append(c)

    selected: List[Dict] = []
    for cell in bins.values():
        cell.sort(key=lambda x: x["_sharp"], reverse=True)
        selected.extend(cell[:max(1, keep_per_cell)])

    selected.sort(key=lambda x: x.get("timestamp", 0))
    if len(selected) > max_images:
        # keep the globally sharpest if we still exceed the cap
        selected = sorted(selected, key=lambda x: x["_sharp"], reverse=True)[:max_images]
        selected.sort(key=lambda x: x.get("timestamp", 0))
    log.info(f"selected {len(selected)} frames "
             f"(dedupe {cell_m:.0f} m grid, keep {keep_per_cell}/cell, cap {max_images})")
    return selected


# ── stage ODM project ──────────────────────────────────────────────────────────
def stage_project(selected: List[Dict], source_dir: Path, out_dir: Path,
                  default_alt: float) -> Tuple[Path, int]:
    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    geo_lines = ["EPSG:4326"]                       # x=lon, y=lat, z=alt (fallback / belt-and-suspenders)
    staged = 0
    exif_ok = 0
    for e in selected:
        src = source_dir / e["file"]
        dst = images_dir / e["file"]
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            log.warning(f"copy failed for {e['file']}: {exc}")
            continue
        lon, lat = e["_lon"], e["_lat"]
        alt = float(e.get("drone_amsl") if e.get("drone_amsl") is not None else default_alt)
        if write_exif_gps(dst, lat, lon, alt):
            exif_ok += 1
        geo_lines.append(f"{e['file']} {lon:.8f} {lat:.8f} {alt:.2f}")
        staged += 1

    (out_dir / "images" / "geo.txt").write_text("\n".join(geo_lines) + "\n")
    if _HAVE_PIEXIF:
        log.info(f"staged {staged} images ({exif_ok} with EXIF GPS) + geo.txt -> {images_dir}")
    else:
        log.warning(f"staged {staged} images + geo.txt -> {images_dir}  "
                    f"(piexif not installed: `pip install piexif` for EXIF GPS; geo.txt used instead)")
    return out_dir, staged


# ── ODM command (Split-Merge + CUDA) ──────────────────────────────────────────
def build_odm_command(out_dir: Path, n_images: int, use_gpu: bool, opt: Dict) -> List[str]:
    project_name = out_dir.name
    cores = max(1, (os.cpu_count() or 4))
    cmd: List[str] = ["docker", "run", "--rm"]
    if use_gpu:
        cmd += ["--gpus", "all"]                     # needs NVIDIA Container Toolkit
    cmd += ["-v", f"{out_dir.resolve()}:/datasets/{project_name}"]
    cmd += [opt["odm_image"] if use_gpu else "opendronemap/odm"]
    cmd += ["--project-path", "/datasets", project_name]
    cmd += [
        "--feature-quality", str(opt["feature_quality"]),
        "--pc-quality", str(opt["pc_quality"]),
        "--max-concurrency", str(cores),
        "--orthophoto-resolution", f"{opt['ortho_res_cm']}",
        "--dsm",
        "--use-exif",                                # plus geo.txt for georeferencing
    ]
    if n_images > opt["chunk_size"]:                 # Split-Merge bounds peak RAM
        cmd += ["--split", str(opt["chunk_size"]),
                "--split-overlap", str(opt["split_overlap"])]
        log.info(f"Split-Merge ON: {n_images} imgs -> ~{opt['chunk_size']}/submodel "
                 f"(overlap {opt['split_overlap']:.0f} m)")
    return cmd


def write_runner(out_dir: Path, cmd: List[str]) -> Path:
    script = out_dir / "run_survey.sh"
    body = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + " \\\n  ".join(cmd) + "\n"
    script.write_text(body)
    script.chmod(0o755)
    log.info(f"wrote runner -> {script}")
    return script


# ── orchestration ──────────────────────────────────────────────────────────────
def run(args) -> int:
    source_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(source_dir)
    if not entries:
        return 1
    selected = select_frames(entries, source_dir,
                             cell_m=args.cell_m, keep_per_cell=args.keep_per_cell,
                             max_images=args.max_images,
                             min_dep=DEFAULTS["min_depression"], min_agl=DEFAULTS["min_agl"],
                             min_sharp=args.min_sharpness, min_conf=args.min_confidence)
    if not selected:
        log.error("no suitable frames after filtering — nothing to process")
        return 1

    opt = dict(DEFAULTS)
    opt["chunk_size"] = args.chunk_size
    out_dir, staged = stage_project(selected, source_dir, out_dir, opt["default_alt_m"])
    if staged == 0:
        return 1

    cmd = build_odm_command(out_dir, staged, use_gpu=args.gpu, opt=opt)
    write_runner(out_dir, cmd)
    log.info("ODM command:\n    " + " ".join(cmd))

    if args.run:
        if shutil.which("docker") is None:
            log.error("docker not found on PATH — cannot --run; use the generated run_survey.sh")
            return 2
        log.info("launching OpenDroneMap (this can take a long time) …")
        try:
            return subprocess.run(cmd).returncode
        except KeyboardInterrupt:
            log.info("interrupted")
            return 130
        except Exception as exc:
            log.error(f"docker run failed: {exc}")
            return 2
    else:
        log.info("prepared only. Launch with:  bash %s   (or re-run with --run)"
                 % (out_dir / "run_survey.sh"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post-flight SfM trigger for OpenDroneMap.")
    p.add_argument("--source", default=str(config.HISTORY_DIR),
                   help="directory with tiles + manifest.json (default: history/)")
    p.add_argument("--out", default=str(_cfg("SURVEY_DIR", config.BASE_DIR / "survey_export")),
                   help="ODM project / export directory (default: survey_export/)")
    p.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"],
                   help="ODM Split-Merge submodel size (default: 300)")
    p.add_argument("--cell-m", type=float, default=DEFAULTS["cell_m"],
                   help="spatial dedupe grid in metres (default: 8)")
    p.add_argument("--keep-per-cell", type=int, default=DEFAULTS["keep_per_cell"],
                   help="sharpest frames kept per grid cell (default: 2)")
    p.add_argument("--max-images", type=int, default=DEFAULTS["max_images"],
                   help="hard cap on selected images (default: 3000)")
    p.add_argument("--min-sharpness", type=float, default=DEFAULTS["min_sharpness"],
                   help="absolute Laplacian-variance blur floor (0 = rank-only, default)")
    p.add_argument("--min-confidence", type=float, default=DEFAULTS["min_confidence"],
                   help="drop frames whose best detection confidence < this (0 = off, default)")
    gpu = p.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", dest="gpu", action="store_true", help="use the CUDA ODM image (default)")
    gpu.add_argument("--no-gpu", dest="gpu", action="store_false", help="use the CPU ODM image")
    p.set_defaults(gpu=True)
    p.add_argument("--run", action="store_true", help="launch the Docker pipeline now")
    return p


if __name__ == "__main__":
    try:
        sys.exit(run(build_parser().parse_args()))
    except Exception as exc:
        log.critical(f"fatal: {exc}")
        sys.exit(1)