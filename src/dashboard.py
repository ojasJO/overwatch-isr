"""
src/dashboard.py — Palantir-grade tactical dashboard (Flask + SocketIO + Leaflet).
=================================================================================
Single-process, threading async mode (no eventlet/gevent). The processing loop
in main.py pushes state via emit_state(); this module broadcasts it to the
browser and serves the historical geo-referenced overlays for the time-scrubber.

Endpoints
  GET /                  the dashboard page (embedded HTML/CSS/JS)
  GET /api/history       JSON manifest of saved geo-referenced overlays
  GET /history/<file>    serves a saved overlay image

Socket events (server -> browser)
  frame_update           per-frame telemetry / detections / frustum / live frame
  history_update         a newly persisted overlay (appears on the timeline)
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict

from flask import (Flask, jsonify, send_from_directory, send_file,
                   make_response, render_template_string)
from flask_socketio import SocketIO

import config

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger("Dashboard")

app = Flask(__name__)
app.config["SECRET_KEY"] = "uav-isr-tactical"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Composite-orthomosaic state (updated by the processing thread via set_mosaic())
_mosaic_meta = {"available": False, "bounds": None, "ts": 0}


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(
        PAGE,
        system_name=config.SYSTEM_NAME,
        classification=config.CLASSIFICATION_LABEL,
        init_lat=30.39395,
        init_lon=74.05953,
        port=config.DASHBOARD_PORT,
    )


@app.route("/api/history")
def api_history():
    try:
        if config.HISTORY_MANIFEST.exists():
            return jsonify(json.loads(config.HISTORY_MANIFEST.read_text()))
    except Exception as exc:
        logger.error(f"/api/history failed: {exc}")
    return jsonify([])


@app.route("/history/<path:fname>")
def history_file(fname: str):
    return send_from_directory(str(config.HISTORY_DIR), fname)


@app.route("/api/mosaic/meta")
def api_mosaic_meta():
    """Bounds + availability of the composite orthomosaic (no image bytes)."""
    return jsonify(_mosaic_meta)


@app.route("/api/mosaic")
def api_mosaic():
    """Serves the composite orthomosaic image itself (cache-busted by ?t=)."""
    path = Path(config.MOSAIC_OUTPUT)
    if not path.exists():
        return ("orthomosaic not yet generated", 404)
    resp = make_response(send_file(str(path), mimetype="image/jpeg", conditional=False))
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


# ── emit helpers (called from the processing thread) ──────────────────────────
def emit_state(state: Dict) -> None:
    try:
        socketio.emit("frame_update", state)
    except Exception as exc:
        logger.debug(f"emit_state failed: {exc}")


def emit_history(entry: Dict) -> None:
    try:
        socketio.emit("history_update", entry)
    except Exception as exc:
        logger.debug(f"emit_history failed: {exc}")


def set_mosaic(bounds: Dict, ts: float = None) -> None:
    """Called by the processing thread after (re)writing the composite mosaic."""
    import time as _t
    if bounds:
        _mosaic_meta["available"] = True
        _mosaic_meta["bounds"] = bounds
        _mosaic_meta["ts"] = int((ts or _t.time()) * 1000)


def emit_mosaic() -> None:
    try:
        socketio.emit("mosaic_update", _mosaic_meta)
    except Exception as exc:
        logger.debug(f"emit_mosaic failed: {exc}")


def start(host: str = None, port: int = None) -> None:
    host = host or config.DASHBOARD_HOST
    port = port or config.DASHBOARD_PORT
    logger.info(f"dashboard at http://localhost:{port}")
    socketio.run(app, host=host, port=port, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)


# ══════════════════════════════════════════════════════════════════════════════
# Embedded frontend
# ══════════════════════════════════════════════════════════════════════════════
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{{ system_name }}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<style>
:root{
  --s0:#04070b; --s1:#080d14; --s2:#0c131d; --s3:#111a26; --s4:#16212f;
  --b1:#172335; --b2:#203049; --b3:#2c425f;
  --t1:#dce7f5; --t2:#7f9cbb; --t3:#415977;
  --green:#15c9a3; --amber:#e8a020; --red:#e0455a; --blue:#3b82f6; --cyan:#22b8e6;
  /* offline-safe stacks: no webfont fetch (edge node may be air-gapped) */
  --mono:'Roboto Mono','SFMono-Regular',ui-monospace,'Consolas','Courier New',monospace;
  --sans:'Inter','Roboto',-apple-system,'Segoe UI','Helvetica Neue',Arial,sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--s0);color:var(--t1);font-family:var(--sans);font-size:12px;-webkit-font-smoothing:antialiased;overflow:hidden}
/* strict monospace for everything that must align in columns */
.mono,.f .v,.mono-coord,.det .co,.det .mg,.det .cf,.hud .v,.tl-time,#clock,.pill .v,.leaflet-popup-content .co{font-family:var(--mono);font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1}
/* sans hierarchy for labels, headers, names */
.brand,.sec-h,.f label,.pill .k,.hud .k,.tl-title,.det .c,#md-txt,.leaflet-popup-content .hd{font-family:var(--sans)}
#app{display:grid;grid-template-rows:20px 36px 1fr 92px;grid-template-columns:300px 1fr 340px;height:100vh}
#classbar{grid-column:1/-1;background:#1a1400;border-bottom:1px solid #3a2900;display:flex;align-items:center;justify-content:center;font-size:10px;letter-spacing:.22em;color:#f0a500;text-transform:uppercase}
#sysbar{grid-column:1/-1;background:var(--s1);border-bottom:1px solid var(--b1);display:flex;align-items:center;padding:0 14px;gap:0}
.brand{font-weight:700;color:var(--green);letter-spacing:.14em;text-transform:uppercase;padding-right:18px;border-right:1px solid var(--b2);margin-right:14px;white-space:nowrap}
.pill{display:flex;align-items:center;gap:7px;padding:0 13px;border-right:1px solid var(--b2);height:100%}
.pill .k{font-size:9px;letter-spacing:.13em;color:var(--t3);text-transform:uppercase}
.pill .v{font-size:11px;color:var(--t1)} .pill .v.ok{color:var(--green)} .pill .v.warn{color:var(--amber)} .pill .v.err{color:var(--red)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);animation:blink 1.7s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.spacer{margin-left:auto;color:var(--t2);font-size:11px;letter-spacing:.06em}
/* panels */
.panel{background:var(--s1);display:flex;flex-direction:column;overflow:hidden}
#left{border-right:1px solid var(--b1)} #right{border-left:1px solid var(--b1)}
.sec{border-bottom:1px solid var(--b1)}
.sec-h{padding:8px 13px 6px;font-size:9px;letter-spacing:.18em;color:var(--t3);text-transform:uppercase;display:flex;align-items:center;gap:7px}
.sec-h .d{width:5px;height:5px;border-radius:50%;background:var(--b3)} .sec-h .d.live{background:var(--green);box-shadow:0 0 5px var(--green)}
.sec-b{padding:2px 13px 12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px 12px}
.f label{font-size:8px;letter-spacing:.14em;color:var(--t3);text-transform:uppercase;display:block;margin-bottom:3px}
.f .v{font-size:15px;color:var(--green);line-height:1} .f .v.dim{color:var(--t2)} .f .v.big{font-size:18px}
.f .v.amber{color:var(--amber)} .f .v.sm{font-size:12px}
.mono-coord{font-size:12px;color:var(--cyan);letter-spacing:.02em;word-break:break-all}
.val{display:flex;align-items:center;gap:10px;padding:9px 13px;margin:0 13px 12px;border:1px solid var(--b2)}
.val .e{font-size:17px} .val.pass{border-color:#15c9a355;background:#15c9a30a} .val.fail{border-color:#e0455a55;background:#e0455a0a}
.val.pass .e{color:var(--green)} .val.fail .e{color:var(--red)} .val .m{font-size:9px;color:var(--t3)}
/* map */
#mapwrap{position:relative;overflow:hidden}
#map{width:100%;height:100%;background:#05080d}
.hud{position:absolute;z-index:600;pointer-events:none;background:rgba(4,7,11,.82);border:1px solid var(--b2);padding:6px 10px;font-size:10px;line-height:1.75}
#hud-tl{top:10px;left:10px} #hud-tr{top:10px;right:10px;text-align:right}
.hud .k{color:var(--t3);font-size:8px;letter-spacing:.13em;text-transform:uppercase}
.hud .v{color:var(--green)} .hud .v.amber{color:var(--amber)}
#mapmode{position:absolute;top:10px;left:50%;transform:translateX(-50%);z-index:600;background:rgba(4,7,11,.9);border:1px solid var(--b2);padding:5px 14px;font-size:9px;letter-spacing:.18em;color:var(--t2);text-transform:uppercase;display:flex;gap:8px;align-items:center}
#mapmode .md{width:8px;height:8px;border-radius:50%;background:var(--t3)}
.maptoggle{position:absolute;bottom:10px;left:10px;z-index:600;display:flex;gap:6px}
.btn{background:var(--s2);border:1px solid var(--b2);color:var(--t2);font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;padding:6px 11px;cursor:pointer}
.btn:hover{border-color:var(--b3);color:var(--t1)} .btn.on{border-color:var(--green);color:var(--green)}
/* right column */
#live{width:100%;display:block;background:#05080d;border-bottom:1px solid var(--b1);aspect-ratio:16/9;object-fit:contain}
#feed{flex:1;overflow-y:auto}
#feed::-webkit-scrollbar{width:3px}#feed::-webkit-scrollbar-thumb{background:var(--b2)}
.det{display:grid;grid-template-columns:54px 1fr 44px;gap:5px;padding:7px 13px;border-bottom:1px solid var(--b1);cursor:pointer}
.det:hover{background:var(--s2)}
.det .c{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--green)}
.det .co{font-size:9px;color:var(--t2);line-height:1.6}
.det .mg{font-size:9px;color:var(--cyan)}
.det .cf{text-align:right;font-size:11px;color:var(--t1)}
#feed-empty{padding:22px 13px;text-align:center;color:var(--t3);font-size:10px;letter-spacing:.1em;text-transform:uppercase}
/* timeline */
#timeline{grid-column:1/-1;background:var(--s1);border-top:1px solid var(--b1);padding:10px 18px;display:flex;flex-direction:column;gap:7px}
.tl-h{display:flex;align-items:center;gap:14px}
.tl-title{font-size:9px;letter-spacing:.18em;color:var(--t3);text-transform:uppercase}
.tl-time{font-size:13px;color:var(--amber);letter-spacing:.04em}
.tl-meta{font-size:9px;color:var(--t2);letter-spacing:.08em;margin-left:auto}
#scrub{width:100%;-webkit-appearance:none;height:4px;background:var(--s4);outline:none;border-radius:2px}
#scrub::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--amber);cursor:pointer;box-shadow:0 0 6px var(--amber)}
#scrub::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--amber);cursor:pointer;border:none}
.tl-controls{display:flex;gap:6px;align-items:center}
.leaflet-popup-content-wrapper{background:linear-gradient(180deg,#0a1118,#06090e);color:var(--t1);border:1px solid var(--b3);border-left:3px solid var(--green);border-radius:2px;box-shadow:0 6px 22px rgba(0,0,0,.6);font-family:var(--sans)}
.leaflet-popup-tip{background:#0a1118;border:1px solid var(--b3)}
.leaflet-popup-content{margin:10px 13px;font-size:11px;line-height:1.65}
.leaflet-popup-content .hd{font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:5px;display:flex;align-items:center;gap:7px}
.leaflet-popup-content .hd .swatch{width:9px;height:9px;border-radius:2px;display:inline-block}
.leaflet-popup-content .kv{display:grid;grid-template-columns:48px 1fr;gap:1px 9px;margin-top:3px}
.leaflet-popup-content .kv .k{color:var(--t3);font-size:9px;letter-spacing:.1em;text-transform:uppercase;font-family:var(--sans);padding-top:2px}
.leaflet-popup-content .co{color:var(--cyan);font-size:11px;letter-spacing:.02em}
.leaflet-popup-content a.leaflet-popup-close-button{color:var(--t2)}
.leaflet-popup-content .badge{margin-top:6px;display:inline-block;font-family:var(--sans);font-size:8px;letter-spacing:.12em;text-transform:uppercase;color:var(--s0);background:var(--green);padding:2px 7px;border-radius:2px;font-weight:700}
.tip{background:rgba(6,9,14,.97)!important;border:1px solid var(--b3)!important;color:var(--t1)!important;font-family:var(--mono)!important;font-size:10px!important;border-radius:2px!important;box-shadow:none!important;padding:3px 7px!important}
</style>
</head>
<body>
<div id="app">
  <div id="classbar">// {{ classification }} //</div>
  <div id="sysbar">
    <div class="brand">{{ system_name }}</div>
    <div class="pill"><span class="dot" id="conn-dot"></span><div><div class="k">Link</div><div class="v ok" id="conn">CONNECTING</div></div></div>
    <div class="pill"><div><div class="k">Source</div><div class="v" id="src">—</div></div></div>
    <div class="pill"><div><div class="k">Platform</div><div class="v" id="pid">—</div></div></div>
    <div class="pill"><div><div class="k">FPS</div><div class="v ok" id="fps">—</div></div></div>
    <div class="pill"><div><div class="k">Frame</div><div class="v" id="frame">—</div></div></div>
    <div class="pill"><div><div class="k">GPU</div><div class="v" id="gpu">—</div></div></div>
    <div class="spacer" id="clock">—</div>
  </div>

  <!-- LEFT: telemetry -->
  <div id="left" class="panel">
    <div class="sec"><div class="sec-h"><span class="d live"></span>Platform Station (static)</div>
      <div class="sec-b">
        <div class="grid2">
          <div class="f"><label>Latitude</label><div class="v sm" id="p-lat">—</div></div>
          <div class="f"><label>Longitude</label><div class="v sm" id="p-lon">—</div></div>
        </div>
        <div class="f" style="margin-top:9px"><label>Station MGRS</label><div class="mono-coord" id="p-mgrs">—</div></div>
      </div></div>
    <div class="sec"><div class="sec-h"><span class="d live"></span>Sensor / Gimbal</div>
      <div class="sec-b"><div class="grid2">
        <div class="f"><label>Altitude AGL</label><div class="v big" id="t-agl">—</div></div>
        <div class="f"><label>True Heading</label><div class="v big" id="t-hdg">—</div></div>
        <div class="f"><label>Depression∠</label><div class="v big amber" id="t-dep">—</div></div>
        <div class="f"><label>Zoom</label><div class="v" id="t-zoom">—</div></div>
        <div class="f"><label>HFOV</label><div class="v dim" id="t-hfov">—</div></div>
        <div class="f"><label>Error Margin</label><div class="v amber" id="t-err">—</div></div>
      </div></div></div>
    <div class="sec"><div class="sec-h"><span class="d"></span>Geo Validation</div>
      <div class="val" id="valbox"><div class="e" id="val-e">—</div>
        <div><div style="font-size:10px;color:var(--t2)" id="val-s">awaiting TGT</div>
        <div class="m" id="val-m">vs HUD target solution</div></div></div>
    </div>
    <div class="sec" style="flex:1"><div class="sec-h"><span class="d"></span>Mission</div>
      <div class="sec-b"><div class="grid2">
        <div class="f"><label>Mode</label><div class="v sm amber" id="t-mode">—</div></div>
        <div class="f"><label>Speed</label><div class="v sm dim" id="t-spd">—</div></div>
        <div class="f"><label>UTC</label><div class="v sm dim" id="t-utc">—</div></div>
        <div class="f"><label>Flight</label><div class="v sm dim" id="t-flt">—</div></div>
        <div class="f"><label>Contacts</label><div class="v sm" id="t-det">0</div></div>
        <div class="f"><label>Tilt conv.</label><div class="v sm dim" id="t-conv">—</div></div>
      </div></div></div>
  </div>

  <!-- CENTER: map -->
  <div id="mapwrap">
    <div id="map"></div>
    <div id="mapmode"><span class="md" id="md"></span><span id="md-txt">NO TRACK</span></div>
    <div class="hud" id="hud-tl">
      <div><span class="k">STATION&nbsp;</span><span class="v" id="h-pos">—</span></div>
      <div><span class="k">AGL&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="v" id="h-agl">—</span></div>
      <div><span class="k">LOS&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="v amber" id="h-los">—</span></div>
    </div>
    <div class="hud" id="hud-tr">
      <div><span class="k">DEP∠&nbsp;</span><span class="v amber" id="h-dep">—</span></div>
      <div><span class="k">HFOV&nbsp;</span><span class="v" id="h-hfov">—</span></div>
      <div><span class="k">ERR&nbsp;&nbsp;</span><span class="v" id="h-err">—</span></div>
    </div>
    <div class="maptoggle">
      <button class="btn on" id="bt-frustum">Frustum</button>
      <button class="btn" id="bt-ortho">Ortho</button>
      <button class="btn on" id="bt-history">History</button>
      <button class="btn" id="bt-follow">Follow</button>
    </div>
  </div>

  <!-- RIGHT: live + feed -->
  <div id="right" class="panel">
    <div class="sec-h"><span class="d live"></span>Live Viewport</div>
    <img id="live" alt="live feed"/>
    <div class="sec-h" style="border-top:1px solid var(--b1)"><span class="d live"></span>Detection Feed <span id="feed-n" style="margin-left:auto;color:var(--t3)">0</span></div>
    <div id="feed"><div id="feed-empty">awaiting detections</div></div>
  </div>

  <!-- BOTTOM: time scrubber -->
  <div id="timeline">
    <div class="tl-h">
      <span class="tl-title">Change-Detection Timeline</span>
      <span class="tl-time" id="tl-time">—</span>
      <div class="tl-controls">
        <button class="btn" id="tl-play">▶ Play</button>
        <button class="btn on" id="tl-now">Live</button>
      </div>
      <span class="tl-meta" id="tl-meta">0 overlays</span>
    </div>
    <input type="range" id="scrub" min="0" max="100" value="100" step="1"/>
  </div>
</div>

<script>
const INIT=[{{ init_lat }},{{ init_lon }}];
const CLR={PERSON:'#15c9a3',BICYCLE:'#22b8e6',VEHICLE:'#e8a020',TRUCK:'#e8a020',BUS:'#e0455a',MOTO:'#a855f7'};
const $=id=>document.getElementById(id);
const set=(id,v)=>{const e=$(id);if(e)e.textContent=v;};

// ---- map ----
const map=L.map('map',{zoomControl:false,attributionControl:false}).setView(INIT,16);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{maxZoom:20,subdomains:'abcd'}).addTo(map);
L.control.zoom({position:'bottomright'}).addTo(map);

const stationIcon=L.divIcon({className:'',iconSize:[26,26],iconAnchor:[13,13],html:
 '<svg width="26" height="26" viewBox="0 0 26 26"><circle cx="13" cy="13" r="11" fill="none" stroke="#15c9a3" stroke-width="1.3"/><circle cx="13" cy="13" r="3.2" fill="#15c9a3"/><line x1="13" y1="1" x2="13" y2="6" stroke="#15c9a3" stroke-width="1.3"/><line x1="13" y1="20" x2="13" y2="25" stroke="#15c9a3" stroke-width="1.3"/><line x1="1" y1="13" x2="6" y2="13" stroke="#15c9a3" stroke-width="1.3"/><line x1="20" y1="13" x2="25" y2="13" stroke="#15c9a3" stroke-width="1.3"/></svg>'});
let station=null, frustum=null, los=null;
let liveDet=L.layerGroup().addTo(map);
let showFrustum=true, showHistory=true, follow=false;

// ---- history overlays for the scrubber ----
let hist=[];           // {ts, layer, rect, dets:[marker], entry}
let tMin=null, tMax=null, scrubbing=false, playing=false, playTimer=null;
let histGroup=L.layerGroup().addTo(map);
let orthoLayer=null, showOrtho=false;   // composite orthomosaic overlay

function addHistory(e){
  if(e.north==null||e.south==null) return;
  const bounds=[[e.south,e.west],[e.north,e.east]];
  const layer=L.imageOverlay('/history/'+e.file,bounds,{opacity:0.55});
  const rect=L.rectangle(bounds,{color:'#22b8e6',weight:0.7,opacity:0.5,fill:false,dashArray:'3 3'});
  const dets=[];
  (e.detections||[]).forEach(d=>{
    if(d.lat==null) return;
    const c=CLR[d.class_name]||'#7f9cbb';
    dets.push(L.circleMarker([d.lat,d.lon],{radius:4,color:c,fillColor:c,fillOpacity:0.25,weight:1})
      .bindTooltip((d.class_name||'?')+' · '+(d.mgrs||''),{className:'tip'}));
  });
  const item={ts:(e.timestamp||0)*1000,layer,rect,dets,entry:e};
  hist.push(item);
  hist.sort((a,b)=>a.ts-b.ts);
  tMin=hist[0].ts; tMax=hist[hist.length-1].ts;
  set('tl-meta',hist.length+' overlays');
  if(!scrubbing) gotoTime(tMax);
}

function applyScrubVisible(cut){
  // cumulative reveal: show overlays up to 'cut'; newest at higher opacity
  let shownTime=tMin;
  hist.forEach(h=>{
    const on=showHistory && h.ts<=cut;
    if(on){ if(!map.hasLayer(h.layer))h.layer.addTo(histGroup); h.rect.addTo(histGroup);
            const age=(cut-h.ts)/Math.max(1,(tMax-tMin));
            h.layer.setOpacity(0.30+0.45*(1-Math.min(1,age)));
            h.dets.forEach(m=>{if(!map.hasLayer(m))m.addTo(histGroup);});
            if(h.ts>shownTime)shownTime=h.ts;
    } else { histGroup.removeLayer(h.layer); histGroup.removeLayer(h.rect); h.dets.forEach(m=>histGroup.removeLayer(m)); }
  });
  const d=new Date(cut);
  set('tl-time', isFinite(cut)? d.toUTCString().slice(17,25)+' UTC':'—');
}

function gotoTime(cut){
  if(tMin==null){return;}
  const pct=tMax>tMin? Math.round((cut-tMin)/(tMax-tMin)*100):100;
  $('scrub').value=pct;
  applyScrubVisible(cut);
}

$('scrub').addEventListener('input',()=>{
  if(tMin==null)return; scrubbing=true; $('tl-now').classList.remove('on');
  const cut=tMin+($('scrub').value/100)*(tMax-tMin);
  applyScrubVisible(cut);
});
$('tl-now').addEventListener('click',()=>{scrubbing=false;$('tl-now').classList.add('on');stopPlay();if(tMax)gotoTime(tMax);});
$('tl-play').addEventListener('click',()=>{ playing?stopPlay():startPlay(); });
function startPlay(){ if(tMin==null)return; playing=true; scrubbing=true; $('tl-now').classList.remove('on'); $('tl-play').textContent='❚❚ Pause'; $('tl-play').classList.add('on');
  let v=Number($('scrub').value); playTimer=setInterval(()=>{ v+=1; if(v>100){v=0;} $('scrub').value=v; applyScrubVisible(tMin+(v/100)*(tMax-tMin)); },120); }
function stopPlay(){ playing=false; clearInterval(playTimer); $('tl-play').textContent='▶ Play'; $('tl-play').classList.remove('on'); }

// ---- composite orthomosaic layer ----
function updateOrtho(meta){
  if(!meta || !meta.available || !meta.bounds) return;
  const b=meta.bounds, bounds=[[b.south,b.west],[b.north,b.east]];
  const url='/api/mosaic?t='+(meta.ts||Date.now());
  if(orthoLayer){ orthoLayer.setUrl(url); orthoLayer.setBounds(bounds); }
  else{ orthoLayer=L.imageOverlay(url,bounds,{opacity:0.92,interactive:false,zIndex:350}); }
  if(showOrtho && !map.hasLayer(orthoLayer)) orthoLayer.addTo(map);
}

// ---- toggles ----
$('bt-frustum').addEventListener('click',()=>{showFrustum=!showFrustum;$('bt-frustum').classList.toggle('on',showFrustum);if(frustum){showFrustum?frustum.addTo(map):map.removeLayer(frustum);}if(los){showFrustum?los.addTo(map):map.removeLayer(los);}});
$('bt-ortho').addEventListener('click',()=>{showOrtho=!showOrtho;$('bt-ortho').classList.toggle('on',showOrtho);
  if(showOrtho){ if(!orthoLayer){fetch('/api/mosaic/meta').then(r=>r.json()).then(updateOrtho);} else {orthoLayer.addTo(map);} }
  else if(orthoLayer){ map.removeLayer(orthoLayer); }});
$('bt-history').addEventListener('click',()=>{showHistory=!showHistory;$('bt-history').classList.toggle('on',showHistory);gotoTime(tMin!=null?(tMin+($('scrub').value/100)*(tMax-tMin)):0);});
$('bt-follow').addEventListener('click',()=>{follow=!follow;$('bt-follow').classList.toggle('on',follow);});

// ---- socket ----
const socket=io();
socket.on('connect',()=>{set('conn','LINKED');$('conn').className='v ok';});
socket.on('disconnect',()=>{set('conn','LINK LOST');$('conn').className='v err';});
socket.on('history_update',e=>addHistory(e));
socket.on('mosaic_update',m=>updateOrtho(m));
socket.on('frame_update',s=>render(s));

function fnum(v,suf='',dp=0){return v==null?'—':(Number(v).toFixed(dp)+suf);}

function render(s){
  const d=s.drone||{}, st=s.stats||{}, v=s.validation||{};
  // sysbar
  set('src',st.video||d.source||'—'); set('pid',d.drone_id||'—');
  set('fps',st.fps!=null?Number(st.fps).toFixed(1):'—'); set('frame',st.frame!=null?st.frame:'—');
  set('gpu',st.gpu_mem_mb?Math.round(st.gpu_mem_mb)+' MB':'—');
  // station (static)
  const p=s.platform;
  if(p){ set('p-lat',p[0].toFixed(6)); set('p-lon',p[1].toFixed(6));
    set('h-pos',p[0].toFixed(4)+', '+p[1].toFixed(4));
    if(!station){station=L.marker(p,{icon:stationIcon}).addTo(map);}else{station.setLatLng(p);} }
  if(d.station_mgrs) set('p-mgrs',d.station_mgrs);
  // telemetry grid
  set('t-agl',fnum(d.agl,' m')); set('t-hdg',fnum(d.heading,'°'));
  set('t-dep',d.depression!=null?d.depression.toFixed(1)+'°':'—');
  set('t-zoom',d.zoom_o!=null?('O '+d.zoom_o+'× / D '+(d.zoom_d??1)+'×'):'—');
  set('t-hfov',fnum(d.hfov,'°',1)); set('t-err',st.error_margin_m!=null?'± '+st.error_margin_m+' m':'—');
  set('t-mode',d.mode||'—'); set('t-spd',fnum(d.speed_kmph,' kmph')); set('t-utc',d.utc||'—'); set('t-flt',d.flt||'—');
  set('t-conv',v.tilt_convention||'—');
  // hud
  set('h-agl',fnum(d.agl,' m AGL')); set('h-dep',d.depression!=null?d.depression.toFixed(1)+'°':'—');
  set('h-hfov',fnum(d.hfov,'°',1)); set('h-err',st.error_margin_m!=null?'± '+st.error_margin_m+' m':'—');
  // validation
  if(v.error_m!=null){const b=$('valbox');b.className='val '+(v.pass?'pass':'fail');
    set('val-e','Δ '+v.error_m.toFixed(1)+' m'); set('val-s',v.pass?'✓ PASS':'✗ FAIL');
    set('val-m','threshold '+(v.threshold_m||200)+' m'+(v.bias_m!=null?' · bias '+v.bias_m+' m':''));}
  // frustum
  if(s.frustum && s.frustum.length===4){
    if(frustum)map.removeLayer(frustum);
    frustum=L.polygon(s.frustum,{color:'#15c9a3',weight:1.2,opacity:0.8,fillColor:'#15c9a3',fillOpacity:0.07});
    if(showFrustum)frustum.addTo(map);
    const cen=[(s.frustum[0][0]+s.frustum[2][0])/2,(s.frustum[0][1]+s.frustum[2][1])/2];
    if(p){ if(los)map.removeLayer(los); los=L.polyline([p,cen],{color:'#e8a020',weight:1,opacity:0.55,dashArray:'5 5'}); if(showFrustum)los.addTo(map);
      const dist=map.distance(p,cen); set('h-los',Math.round(dist)+' m'); }
    if(follow && p) map.panTo(cen,{animate:true,duration:0.4});
  }
  // MASTER TRACKS — one de-duplicated, averaged pin per physical object
  // (collapses parallax-scattered repeats; no 15 dots for 1 truck).
  liveDet.clearLayers();
  const tracks = (s.tracks && s.tracks.length) ? s.tracks : (s.detections || []);
  const live = s.detections || [];
  set('t-det', tracks.length); set('feed-n', tracks.length);
  const rows = [];
  tracks.forEach(tk => {
    const c = CLR[tk.class_name] || '#7f9cbb';
    const hasGeo = (tk.lat != null && tk.lon != null);
    const conf = ((tk.confidence || 0) * 100).toFixed(0);
    const n = tk.count || 1;

    if (hasGeo) {
      const popupHtml =
        `<div class="hd"><span class="swatch" style="background:${c}"></span>${tk.class_name || '?'}</div>` +
        `<div class="kv">` +
          `<span class="k">MGRS</span><span class="co">${tk.mgrs || '—'}</span>` +
          `<span class="k">Lat</span><span class="co">${tk.lat.toFixed(6)}</span>` +
          `<span class="k">Lon</span><span class="co">${tk.lon.toFixed(6)}</span>` +
          `<span class="k">Conf</span><span class="co">${conf}%</span>` +
          `<span class="k">Obs</span><span class="co">${n}</span>` +
        `</div><span class="badge">MASTER TRACK</span>`;
      // confirmation grows the pin slightly with observation count (capped)
      const r = 6 + Math.min(6, Math.log2(n + 1) * 2);
      L.circleMarker([tk.lat, tk.lon], {
        radius: r, color: c, weight: 2, fillColor: c, fillOpacity: 0.30
      }).bindPopup(popupHtml).bindTooltip(
        `${tk.class_name || '?'} · ${tk.mgrs || ''}`, {className: 'tip', direction: 'top'}
      ).addTo(liveDet);
    }

    const coords = hasGeo ? `${tk.lat.toFixed(5)}, ${tk.lon.toFixed(5)}` : 'NO GEO';
    const row = document.createElement('div');
    row.className = 'det';
    row.innerHTML =
      `<div class="c" style="color:${c}">${tk.class_name || '?'}<span style="color:var(--t3)"> ×${n}</span></div>` +
      `<div><div class="co">${coords}</div><div class="mg">${tk.mgrs || ''}</div></div>` +
      `<div class="cf">${conf}%</div>`;
    if (hasGeo) row.addEventListener('click', () => map.flyTo([tk.lat, tk.lon], 18));
    rows.push(row);
  });

  const feed = $('feed');
  feed.innerHTML = '';
  if (!tracks.length) {
    feed.innerHTML = '<div id="feed-empty">no confirmed contacts</div>';
  } else {
    const frag = document.createDocumentFragment();
    rows.forEach(r => frag.appendChild(r));
    feed.appendChild(frag);
  }
  // map mode banner — driven by confirmed master tracks
  const md=$('md'), mt=$('md-txt'), nT=tracks.length;
  if(!nT){md.style.cssText='background:var(--t3)';mt.textContent='NO TRACK';}
  else if(nT<=3){md.style.cssText='background:var(--amber);box-shadow:0 0 6px var(--amber)';mt.textContent='MONITOR · '+nT+' CONTACTS';}
  else{md.style.cssText='background:var(--red);box-shadow:0 0 6px var(--red)';mt.textContent='ALERT · '+nT+' CONTACTS';}
  // live viewport (raw current-frame boxes shown in the image itself)
  if(s.frame_jpeg){ $('live').src='data:image/jpeg;base64,'+s.frame_jpeg; }
}

// clock + initial history load
setInterval(()=>set('clock',new Date().toUTCString().slice(5,25)+' UTC'),1000);
fetch('/api/history').then(r=>r.json()).then(list=>{ (list||[]).forEach(addHistory); }).catch(()=>{});
fetch('/api/mosaic/meta').then(r=>r.json()).then(updateOrtho).catch(()=>{});
</script>
</body>
</html>"""