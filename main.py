"""
main.py — Offline UAV ISR pipeline orchestrator (single-PC monolith).
=====================================================================
Boot sequence:
  1. cleanup_workspace()  — ensure the canonical tree exists; archive any
     obsolete Phase-1 proof-of-concept files into ./archive (never delete
     canonical files).
  2. start the dashboard in a daemon thread (Flask + SocketIO).
  3. discover every video in ./footage, sort chronologically.
  4. process each video sequentially: decode -> OCR telemetry -> detect ->
     geolocate (with platform isolation + smoothing) -> push state to the UI ->
     persist high-value geo-referenced frames to ./history.
  5. keep the dashboard alive after processing so the operator can scrub the
     change-detection timeline.

Run:  python main.py    then open  http://localhost:5000
"""
from __future__ import annotations
import sys
import time
import shutil
import base64
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2

import config
from src.geolocator import Geolocator, latlon_to_mgrs, MasterTrackRegistry
from src.video_processor import VideoProcessor, HistoryWriter
from src import dashboard

# Stitcher is optional; if it (or OpenCV) fails to import, the pipeline still runs.
try:
    from src.stitcher import OrthoStitcher
except Exception as _exc:                      # pragma: no cover
    OrthoStitcher = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-15s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# Detector is heavy (torch + CUDA). Imported here but instantiated in a guard so
# the rest of the system still boots if the CUDA stack or weights are absent.
try:
    from src.detector import Detector
except Exception as exc:                       # pragma: no cover
    Detector = None
    log.warning(f"detector import failed ({exc}); detection will be disabled")


# ─────────────────────────────────────────────────────────────────────────────
# 1. workspace cleanup
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_workspace() -> None:
    # HARD PURGE: wipe all prior-run map state so every boot starts pristine.
    # Removes ./history (geo-referenced tiles + manifest + orthomosaic.jpg), any
    # ./mosaic dir, and a stray orthomosaic.jpg outside ./history.
    purge_targets = [config.HISTORY_DIR, config.BASE_DIR / "mosaic"]
    for tgt in purge_targets:
        try:
            if tgt.exists():
                shutil.rmtree(tgt, ignore_errors=True)
                log.info(f"purged {tgt}")
        except Exception as exc:
            log.error(f"could not purge {tgt}: {exc}")
    for stray in {config.MOSAIC_OUTPUT, config.BASE_DIR / "orthomosaic.jpg"}:
        try:
            Path(stray).unlink(missing_ok=True)
        except Exception:
            pass

    # (re)create the canonical runtime directories
    for d in (config.FOOTAGE_DIR, config.HISTORY_DIR, config.MODELS_DIR, config.SRC_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            log.error(f"could not create {d}: {exc}")

    # archive obsolete proof-of-concept artefacts (never the canonical files)
    canonical = {
        "config.py", "main.py", "requirements.txt",
        "src/__init__.py", "src/video_processor.py", "src/geolocator.py",
        "src/detector.py", "src/dashboard.py", "src/stitcher.py", "src/survey_processor.py",
    }
    archived = 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest_root = config.ARCHIVE_DIR / f"phase1_{stamp}"
    for rel in config.OBSOLETE_ARTIFACTS:
        if rel in canonical:
            continue
        src = config.BASE_DIR / rel
        if not src.exists():
            continue
        try:
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            archived += 1
            log.info(f"archived obsolete artifact: {rel}")
        except Exception as exc:
            log.error(f"could not archive {rel}: {exc}")
    log.info(f"workspace purged + clean · {archived} obsolete artifact(s) archived"
             + (f" -> {dest_root}" if archived else ""))


# ─────────────────────────────────────────────────────────────────────────────
# 2. footage discovery
# ─────────────────────────────────────────────────────────────────────────────
def discover_footage() -> List[Path]:
    try:
        vids = [p for p in config.FOOTAGE_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS]
    except Exception as exc:
        log.error(f"could not read footage dir: {exc}")
        return []
    # chronological: file mtime first, then name (Asteria stamps date/time in the name)
    vids.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return vids


# ─────────────────────────────────────────────────────────────────────────────
# state assembly + live frame encoding
# ─────────────────────────────────────────────────────────────────────────────
def _encode_jpeg(frame) -> Optional[str]:
    try:
        img = frame
        if img.shape[1] > config.LIVE_JPEG_MAX_W:
            s = config.LIVE_JPEG_MAX_W / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), config.LIVE_JPEG_QUALITY])
        if ok:
            return base64.b64encode(buf).decode("ascii")
    except Exception as exc:
        log.debug(f"jpeg encode failed: {exc}")
    return None


def build_state(geo: Geolocator, telem: Dict, dets: List[Dict], frame_jpeg: Optional[str],
                stats: Dict, tracks: Optional[List[Dict]] = None) -> Dict:
    plat = geo.platform_point()
    station_mgrs = latlon_to_mgrs(plat[0], plat[1], config.MGRS_PRECISION) if plat else ""
    drone = {
        "lat": telem.get("lat"), "lon": telem.get("lon"),
        "agl": telem.get("agl"), "amsl": telem.get("amsl"),
        "heading": telem.get("heading"), "tilt": telem.get("tilt"), "pan": telem.get("pan"),
        "hfov": telem.get("hfov"), "zoom_o": telem.get("zoom_o"), "zoom_d": telem.get("zoom_d"),
        "depression": geo.depression_deg(telem), "mode": telem.get("mode"),
        "drone_id": telem.get("drone_id"), "utc": telem.get("utc"), "flt": telem.get("flt"),
        "source": telem.get("source"), "speed_kmph": telem.get("speed_kmph"),
        "station_mgrs": station_mgrs,
    }
    feed = [{
        "class_name": d.get("class_name"), "confidence": d.get("confidence"),
        "lat": d.get("lat"), "lon": d.get("lon"), "mgrs": d.get("mgrs"),
        "accuracy_m": d.get("accuracy_m"), "track_id": d.get("track_id"),
        "timestamp": telem.get("timestamp"),
    } for d in dets[:config.MAX_FEED_DETECTIONS]]
    return {
        "drone": drone,
        "platform": list(plat) if plat else None,
        "detections": feed,
        "tracks": tracks or [],          # de-duplicated Master Tracks (one pin per object)
        "frustum": geo.frustum(telem),
        "validation": geo.validate(telem),
        "stats": stats,
        "frame_jpeg": frame_jpeg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. per-video processing
# ─────────────────────────────────────────────────────────────────────────────
def process_video(path: Path, detector, history: HistoryWriter, stitcher,
                  registry: MasterTrackRegistry,
                  video_idx: int, video_total: int) -> None:
    log.info(f"━━ [{video_idx}/{video_total}] {path.name} ━━")
    try:
        vp = VideoProcessor(path)
    except Exception as exc:
        log.error(f"cannot open {path.name}: {exc}")
        return

    geo = Geolocator(vp.width, vp.height)   # fresh isolation/calibration per sortie
    emitted = 0
    t_start = time.time()
    last_log = t_start
    last_mosaic = t_start

    try:
        for fidx, t_sec, frame in vp.frames(config.FRAME_STRIDE):
            try:
                telem = vp.parse_overlay(frame, fidx, t_sec)
                geo.update_platform(telem)

                dets: List[Dict] = []
                if detector is not None:
                    dets = detector.detect(frame, hfov=telem.get("hfov"))
                    dets = geo.locate(dets, telem)
                    registry.observe(dets, time.time())   # collapse repeats -> Master Tracks

                # live viewport (throttled): annotate over the raw frame
                frame_jpeg = None
                if emitted % config.LIVE_EMIT_EVERY_N == 0:
                    shown = detector.annotate(frame, dets) if detector is not None else frame
                    frame_jpeg = _encode_jpeg(shown)

                now = time.time()
                fps = (emitted + 1) / max(1e-3, now - t_start)
                stats = {
                    "fps": round(fps, 1), "frame": fidx, "total_frames": vp.frame_count,
                    "det_count": len(dets),
                    "gpu_mem_mb": detector.gpu_mem_mb() if detector is not None else 0,
                    "inf_ms": round(detector.avg_inf_ms, 1) if detector is not None else 0,
                    "error_margin_m": geo.boresight_error_m(telem),
                    "ocr_ok": vp.ocr_stats["ok"], "ocr_fail": vp.ocr_stats["fail"],
                    "video": path.name, "video_idx": video_idx, "video_total": video_total,
                }
                dashboard.emit_state(build_state(geo, telem, dets, frame_jpeg, stats,
                                                 tracks=registry.tracks()))
                emitted += 1

                # persist a geo-referenced overlay for the mosaic / scrubber — ONLY for
                # steeply-downward frames. Oblique/horizon shots produce a degenerate
                # homography and a horizon-stretched overlay bbox, so they bypass both
                # the stitcher and the history saver to keep the map pristine.
                depression = geo.depression_deg(telem)
                has_det = len(dets) > 0
                if (history.should_save(now, has_det)
                        and depression is not None
                        and depression >= config.STITCH_MIN_DEPRESSION_DEG):
                    save_frame = frame
                    bbox = geo.geo_bbox(telem)
                    if stitcher is not None:
                        # ortho-rectify to a north-up tile + fold into the master mosaic;
                        # geo supplies the DEM-corrected footprint; the stitcher's own
                        # tile bounds match the returned imagery
                        save_frame = stitcher.add_frame(frame, telem, geo=geo)
                        if stitcher.last_tile_bounds is not None:
                            bbox = stitcher.last_tile_bounds
                    if bbox is not None:
                        plat = geo.platform_point()
                        meta = {
                            "timestamp": now, "utc": telem.get("utc", ""),
                            "source_file": path.name,
                            "drone_lat": plat[0] if plat else None,
                            "drone_lon": plat[1] if plat else None,
                            "detections": [{
                                "class_name": d.get("class_name"), "confidence": d.get("confidence"),
                                "lat": d.get("lat"), "lon": d.get("lon"), "mgrs": d.get("mgrs"),
                                "timestamp": now,
                            } for d in dets if d.get("lat") is not None],
                        }
                        entry = history.save(save_frame, bbox, meta, now)
                        if entry is not None:
                            dashboard.emit_history(entry)

                if now - last_log >= 4.0:
                    log.info(f"  {path.name}: frame {fidx}/{vp.frame_count} · "
                             f"{fps:.1f} fps · {len(dets)} contacts · "
                             f"err ±{stats['error_margin_m']}m")
                    last_log = now

                # periodically refresh the composite orthomosaic on the dashboard
                if (stitcher is not None and stitcher.mosaic_bounds is not None
                        and now - last_mosaic >= config.STITCH_MOSAIC_REFRESH_S):
                    if stitcher.save_mosaic(config.MOSAIC_OUTPUT):
                        dashboard.set_mosaic(stitcher.mosaic_bounds, now)
                        dashboard.emit_mosaic()
                    last_mosaic = now
            except Exception as exc:
                log.error(f"frame {fidx} failed: {exc}")
                continue
    finally:
        vp.close()
    log.info(f"✓ {path.name} complete · {emitted} frames processed")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 64)
    log.info(f"{config.SYSTEM_NAME} · offline ISR pipeline")
    log.info("=" * 64)

    cleanup_workspace()

    # boot dashboard in the background
    th = threading.Thread(target=dashboard.start, daemon=True, name="dashboard")
    th.start()
    time.sleep(1.2)
    log.info(f"dashboard live → http://localhost:{config.DASHBOARD_PORT}")

    videos = discover_footage()
    if not videos:
        log.warning(f"no video files in {config.FOOTAGE_DIR} "
                    f"(looking for {', '.join(config.VIDEO_EXTENSIONS)})")
        log.warning("drop a recorded .avi into ./footage and restart. "
                    "dashboard will stay up.")
    else:
        log.info(f"discovered {len(videos)} video(s): {', '.join(v.name for v in videos)}")

    detector = None
    if Detector is not None and videos:
        try:
            detector = Detector()
        except Exception as exc:
            log.error(f"detector unavailable ({exc}); continuing without detection")
            detector = None

    history = HistoryWriter()
    registry = MasterTrackRegistry()        # persists across all sorties in this run
    stitcher = None
    if OrthoStitcher is not None and videos:
        try:
            stitcher = OrthoStitcher()
            log.info(f"orthomosaic stitcher ready · {stitcher.stats}")
        except Exception as exc:
            log.error(f"stitcher unavailable ({exc}); saving raw frames instead")
            stitcher = None

    try:
        for i, vid in enumerate(videos, 1):
            process_video(vid, detector, history, stitcher, registry, i, len(videos))
        log.info(f"master tracks consolidated: {registry.n_tracks}")
        if stitcher is not None:
            stitcher.save_mosaic(config.MOSAIC_OUTPUT)
            if stitcher.mosaic_bounds is not None:
                dashboard.set_mosaic(stitcher.mosaic_bounds)
                dashboard.emit_mosaic()
        if videos:
            log.info("=" * 64)
            log.info("all footage processed · dashboard remains live for review")
            log.info("scrub the timeline to inspect change-detection over time")
            log.info("=" * 64)
        while True:                              # keep the UI alive for the operator
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("shutdown requested — exiting")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical(f"fatal: {exc}")
        sys.exit(1)