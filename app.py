"""
PixelConductor – Flask Backend
==============================
Serves PixelConductor.html and provides deep audio analysis via:
  - librosa  : beat tracking (DBN), onset detection, spectral analysis, section segmentation
  - demucs   : htdemucs_6s – 6-stem GPU separation (drums/bass/vocals/guitar/piano/other)
  - HPSS     : drums split → kick/snare + cymbals
  - 2nd-pass : htdemucs on the 'other' stem → bells / choir / strings / pad sub-stems
  - scipy    : sub-bass frequency band extraction
  - matplotlib: mel spectrogram PNGs per stem

Run:
    python app.py
Then open: http://localhost:7842
"""

import os
import re
import json
import uuid
import time
import shutil
import threading
import queue
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import Flask, send_file, jsonify, request, Response, send_from_directory
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────
HERE            = Path(__file__).parent
HTML_FILE       = HERE / "PixelConductor.html"
DRAFTS_DIR      = HERE / "drafts"
XLIGHTS_AUDIO   = Path(os.environ.get("PC_AUDIO_DIR", r"E:\Code\xlights\audio"))
XLIGHTS_SHOW_DIR= Path(os.environ.get("PC_SHOW_DIR",  r"E:\Code\xlights"))
PORT            = 7842

# Primary stems from demucs htdemucs_6s
STEMS = ["drums", "bass", "vocals", "guitar", "piano", "other"]

# Derived stems computed after primary separation
DERIVED_STEMS = [
    "drums_perc",    # HPSS percussive of drums  → kick, snare
    "drums_harm",    # HPSS harmonic of drums    → cymbals, hi-hats
    "other_bells",   # 2nd-pass demucs: other→drums  → bells, glockenspiel
    "other_choir",   # 2nd-pass demucs: other→vocals → choir, backing harmonies
    "other_strings", # 2nd-pass demucs: other→other  → strings, pads, synths
    "band_sub",      # Low-pass <80 Hz of master mix → sub-bass pulse
]

ALL_STEMS = STEMS + DERIVED_STEMS

STEM_COLORS = {
    # Primary
    "drums":         "#e74c3c",
    "bass":          "#3498db",
    "vocals":        "#2ecc71",
    "guitar":        "#9b59b6",
    "piano":         "#f39c12",
    "other":         "#1abc9c",
    # Derived
    "drums_perc":    "#c0392b",   # dark red   – kick/snare transients
    "drums_harm":    "#ff7675",   # light pink – cymbals/hi-hats shimmer
    "other_bells":   "#00cec9",   # cyan       – bells & glockenspiel
    "other_choir":   "#6c5ce7",   # indigo     – choir & backing vocals
    "other_strings": "#a29bfe",   # lavender   – strings, pads, synths
    "band_sub":      "#74b9ff",   # sky blue   – sub-bass energy
}

WAVEFORM_BINS  = 16000
SPEC_TIME_BINS = 2000
SPEC_MEL_BINS  = 128

# Onset detection tuning per stem type
# (delta controls sensitivity – lower = more onsets detected)
ONSET_DELTA = {
    "drums":         0.04,   # very sensitive
    "drums_perc":    0.03,   # most sensitive – every kick/snare
    "drums_harm":    0.05,
    "bass":          0.07,
    "vocals":        0.06,
    "guitar":        0.05,
    "piano":         0.05,
    "other":         0.07,
    "other_bells":   0.03,   # bells have sharp transients
    "other_choir":   0.10,   # choir sustains – fewer onsets
    "other_strings": 0.10,   # strings sustain
    "band_sub":      0.08,
}

app = Flask(__name__)
CORS(app)

# job_id → {"status", "progress", "message", "result_path", "spec_dir", "error", "queue"}
JOBS: dict[str, dict] = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(HTML_FILE)

@app.route("/PixelConductor.html")
def html_explicit():
    return send_file(HTML_FILE)

@app.route("/api/browse")
def browse():
    """List audio files in the xLights audio directory."""
    files = []
    if XLIGHTS_AUDIO.exists():
        for p in sorted(XLIGHTS_AUDIO.iterdir()):
            if p.suffix.lower() in (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"):
                analyzed = p.with_suffix(".analysis.json").exists()
                files.append({"name": p.name, "path": str(p), "size": p.stat().st_size, "analyzed": analyzed})
    return jsonify(files)

@app.route("/api/browse/xsq")
def browse_xsq():
    """List .xsq files in the xLights project folder (parent of audio dir)."""
    xsq_dir = XLIGHTS_AUDIO.parent
    files = []
    if xsq_dir.exists():
        for p in sorted(xsq_dir.glob("*.xsq"), key=_xsq_sort_key):
            files.append({"name": p.name, "path": str(p), "size": p.stat().st_size})
    return jsonify(files)

@app.route("/api/xsq")
def serve_xsq():
    path = request.args.get("path", "")
    p = Path(path)
    if not p.exists() or p.suffix.lower() != ".xsq":
        return jsonify({"error": "File not found"}), 404
    return Response(p.read_text(encoding="utf-8"), mimetype="application/xml")

@app.route("/api/xsq-dir")
def xsq_dir():
    """Folder that Open XSQ lists and Save As writes into."""
    return jsonify({"dir": str(XLIGHTS_AUDIO.parent)})

# One save/draft write at a time: concurrent requests used to race on a shared
# .tmp file and fail on Windows.
SAVE_LOCK = threading.Lock()

# Backup retention. Auto-save runs every 30 s, so "keep the last N" alone can
# wipe out the pre-session version within minutes. Instead keep the newest
# BACKUP_KEEP_RECENT, plus the OLDEST backup of every hour (for a week) and of
# every day (for 90 days) — the oldest in a bucket is the state before that
# stretch of work began.
BACKUP_KEEP_RECENT = 20
BACKUP_HOURLY_DAYS = 7
BACKUP_DAILY_DAYS  = 90

def _backup_time(name: str, stem: str):
    m = re.fullmatch(re.escape(stem) + r"_(\d{8})_(\d{6})(?:_(\d{3}))?\.xsq", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None

def _backup_xsq(p: Path):
    """Copy p into .pc_backups (skipping exact duplicates of the newest backup), then prune."""
    backup_dir = p.parent / ".pc_backups"
    backup_dir.mkdir(exist_ok=True)
    entries = sorted((t, b) for b in backup_dir.glob(f"{p.stem}_*.xsq")
                     if (t := _backup_time(b.name, p.stem)))
    current = p.read_bytes()
    if not entries or entries[-1][1].read_bytes() != current:
        now = datetime.now()
        bak = backup_dir / f"{p.stem}_{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}.xsq"
        shutil.copy2(p, bak)
        entries.append((now, bak))

    now  = datetime.now()
    keep = {b for _, b in entries[-BACKUP_KEEP_RECENT:]}
    hours, days = set(), set()
    for t, b in entries:                       # oldest first
        age = now - t
        if age <= timedelta(days=BACKUP_HOURLY_DAYS) and t.strftime("%Y%m%d%H") not in hours:
            hours.add(t.strftime("%Y%m%d%H")); keep.add(b)
        if age <= timedelta(days=BACKUP_DAILY_DAYS) and t.strftime("%Y%m%d") not in days:
            days.add(t.strftime("%Y%m%d")); keep.add(b)
    for _, b in entries:
        if b not in keep:
            try: b.unlink()
            except OSError: pass

def _atomic_write(p: Path, text: str):
    """Write via a unique temp file + rename so readers never see a partial file.
    Retries the rename briefly: on Windows it fails while xLights/antivirus has the file open."""
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(tmp, p)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.2)
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass

@app.route("/api/save-xsq", methods=["POST"])
def save_xsq():
    data = request.get_json() or {}
    path = data.get("path", "")
    xml  = data.get("xml", "")
    p = Path(path)
    if not path or p.suffix.lower() != ".xsq":
        return jsonify({"error": "Invalid path"}), 400
    if not p.parent.exists():
        return jsonify({"error": f"Folder not found: {p.parent}"}), 400

    new_effect_count = len(re.findall(r'<Effect\b', xml))

    with SAVE_LOCK:
        if p.exists():
            try:
                _backup_xsq(p)
            except Exception as e:
                # Never overwrite a file we couldn't back up
                return jsonify({"error": f"Backup failed, file not saved: {e}"}), 500
        try:
            _atomic_write(p, xml)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "path": str(p), "effects": new_effect_count})

# ── Drafts ────────────────────────────────────────────────────────────────────
# The editor posts its working sequence here every few seconds while it has
# unsaved changes, so a reload, crash or discarded tab can't lose work.
# drafts/<id>.xsq + <id>.json; discarded drafts move to drafts/discarded/.

def _draft_id(name: str) -> str:
    base = Path(name or "untitled").name
    if base.lower().endswith(".xsq"):
        base = base[:-4]
    return re.sub(r"[^\w.\- ]", "_", base).strip(" .") or "untitled"

@app.route("/api/drafts")
def list_drafts():
    drafts = []
    if DRAFTS_DIR.exists():
        for meta in DRAFTS_DIR.glob("*.json"):
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue
            fp = Path(d["path"]) if d.get("path") else None
            d["fileMtime"] = fp.stat().st_mtime if fp and fp.exists() else None
            drafts.append(d)
    drafts.sort(key=lambda d: d.get("savedAt", 0), reverse=True)
    return jsonify(drafts)

@app.route("/api/draft", methods=["GET"])
def get_draft():
    f = DRAFTS_DIR / f"{_draft_id(request.args.get('id', ''))}.xsq"
    if not f.exists():
        return jsonify({"error": "No such draft"}), 404
    return Response(f.read_text(encoding="utf-8"), mimetype="application/xml")

@app.route("/api/draft", methods=["POST"])
def put_draft():
    data = request.get_json() or {}
    xml  = data.get("xml", "")
    if not xml:
        return jsonify({"error": "Empty draft"}), 400
    did  = _draft_id(data.get("name", ""))
    meta = {"id": did, "name": data.get("name") or "untitled", "path": data.get("path") or None,
            "effects": data.get("effects", 0), "savedAt": time.time()}
    with SAVE_LOCK:
        DRAFTS_DIR.mkdir(exist_ok=True)
        _atomic_write(DRAFTS_DIR / f"{did}.xsq", xml)
        _atomic_write(DRAFTS_DIR / f"{did}.json", json.dumps(meta, indent=2))
    return jsonify({"ok": True, "id": did})

@app.route("/api/draft/discard", methods=["POST"])
def discard_draft():
    """archive=true (user discarded it) moves the draft to drafts/discarded/;
    archive=false (its content was just saved to the real file) deletes it."""
    data = request.get_json() or {}
    did  = _draft_id(data.get("id") or data.get("name") or "")
    with SAVE_LOCK:
        files = [f for f in (DRAFTS_DIR / f"{did}.xsq", DRAFTS_DIR / f"{did}.json") if f.exists()]
        if data.get("archive") and files:
            arch = DRAFTS_DIR / "discarded"
            arch.mkdir(exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            for f in files:
                shutil.move(str(f), str(arch / f"{f.stem}_{ts}{f.suffix}"))
        else:
            for f in files:
                f.unlink()
    return jsonify({"ok": True})

@app.route("/api/analyze/cached")
def get_cached_analysis():
    audio_path = request.args.get("path", "")
    json_path  = Path(audio_path).with_suffix(".analysis.json")
    if not json_path.exists():
        return jsonify({"error": "No cache"}), 404
    with open(json_path, encoding="utf-8") as f:
        return Response(f.read(), mimetype="application/json")

@app.route("/api/analyze/clear", methods=["POST"])
def clear_cached_analysis():
    data       = request.get_json() or {}
    audio_path = data.get("path", "")
    json_path  = Path(audio_path).with_suffix(".analysis.json")
    if json_path.exists():
        json_path.unlink()
    return jsonify({"ok": True})

@app.route("/api/audio")
def serve_audio():
    path = request.args.get("path", "")
    p = Path(path)
    if not p.exists() or not p.is_file():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(p), mimetype="audio/mpeg", conditional=True)

@app.route("/api/show-layout")
def show_layout():
    """Parse xlights_rgbeffects.xml and return background + all model positions/shapes."""
    import xml.etree.ElementTree as ET
    xml_path = XLIGHTS_SHOW_DIR / "xlights_rgbeffects.xml"
    if not xml_path.exists():
        return jsonify({"error": "xlights_rgbeffects.xml not found"}), 404

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError as e:
        return jsonify({"error": f"XML parse error: {e}"}), 500

    settings = {}
    settings_el = root.find('settings')
    if settings_el is not None:
        for child in settings_el:
            settings[child.tag] = child.get('value', '')

    bg_image  = settings.get('backgroundImage', '')
    bg_bright = int(settings.get('backgroundBrightness', '100'))
    prev_w    = int(settings.get('previewWidth',  '1280'))
    prev_h    = int(settings.get('previewHeight', '720'))
    bg_scale  = int(settings.get('scaleImage', '1'))

    def parse_submodels(model_el):
        """Return [{name, nodes:[1-based node indices], layout}] for a model's
        <subModel> children. Node ranges live in line0/line1/... attributes as
        comma-separated values or 'a-b' ranges (e.g. line0="1,17-31")."""
        subs = []
        for sm in model_el.findall('subModel'):
            # gather lineN attributes in numeric order
            lines = sorted(
                ((int(k[4:]), v) for k, v in sm.attrib.items()
                 if k.startswith('line') and k[4:].isdigit()),
                key=lambda kv: kv[0])
            nodes = []
            for _, spec in lines:
                for part in spec.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    if '-' in part:
                        a, b = part.split('-', 1)
                        try:
                            nodes.extend(range(int(a), int(b) + 1))
                        except ValueError:
                            pass
                    elif part.isdigit():
                        nodes.append(int(part))
            subs.append({
                "name":   sm.get('name', ''),
                "nodes":  nodes,
                "layout": sm.get('layout', 'horizontal'),
            })
        return subs

    models = []
    for m in root.findall('.//models/model'):
        models.append({
            "name":           m.get('name', ''),
            "displayAs":      m.get('DisplayAs', ''),
            "worldX":         float(m.get('WorldPosX', 0)),
            "worldY":         float(m.get('WorldPosY', 0)),
            "scaleX":         float(m.get('ScaleX', 1)),
            "scaleY":         float(m.get('ScaleY', 1)),
            "gridW":          int(m.get('CustomWidth',  0)),
            "gridH":          int(m.get('CustomHeight', 0)),
            "compressed":     m.get('CustomModelCompressed', ''),
            "x2":             float(m.get('X2', 0)),
            "y2":             float(m.get('Y2', 0)),
            "arc":            float(m.get('Arc', 180)),
            "nodesPerArch":   int(m.get('NodesPerArch', 0)),
            "numArches":      int(m.get('NumArches', 1)),
            "nodesPerString": int(m.get('NodesPerString', 0)),
            "numStrings":     int(m.get('NumStrings', 1)),
            "dropPattern":    m.get('DropPattern', ''),
            "height":         float(m.get('Height', 0)),
            "subModels":      parse_submodels(m),
        })

    groups = []
    for g in root.findall('.//modelGroups/modelGroup'):
        members = [x.strip() for x in g.get('models', '').split(',') if x.strip()]
        groups.append({"name": g.get('name', ''), "members": members})

    return jsonify({
        "backgroundImage":      bg_image,
        "backgroundBrightness": bg_bright,
        "backgroundScale":      bg_scale,
        "previewWidth":         prev_w,
        "previewHeight":        prev_h,
        "models":               models,
        "groups":               groups,
    })


@app.route("/api/image")
def serve_image():
    path = request.args.get("path", "")
    p = Path(path)
    if not p.exists() or not p.is_file():
        return jsonify({"error": "File not found"}), 404
    suffix = p.suffix.lower()
    mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp'}
    mime = mime_map.get(suffix, 'image/jpeg')
    return send_file(str(p), mimetype=mime, conditional=True)


@app.route("/api/analyze", methods=["POST"])
def start_analysis():
    data = request.get_json()
    audio_path = data.get("path", "")
    if not Path(audio_path).exists():
        return jsonify({"error": "File not found"}), 400

    job_id = uuid.uuid4().hex[:12]
    q = queue.Queue()
    JOBS[job_id] = {
        "status": "pending", "progress": 0, "message": "Queued",
        "result_path": None, "spec_dir": None, "error": None, "queue": q
    }

    t = threading.Thread(target=_run_analysis, args=(job_id, audio_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})

@app.route("/api/analyze/<job_id>/stream")
def stream_progress(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Unknown job"}), 404

    def generate():
        job = JOBS[job_id]
        q   = job["queue"]
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("stage") in ("done", "error"):
                    break
            except queue.Empty:
                yield "data: {\"stage\":\"ping\"}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/analyze/<job_id>/result")
def get_result(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Unknown job"}), 404
    job = JOBS[job_id]
    if job["status"] != "done":
        return jsonify({"error": "Not ready", "status": job["status"]}), 202
    with open(job["result_path"], encoding="utf-8") as f:
        return Response(f.read(), mimetype="application/json")

@app.route("/api/spectrogram/<job_id>/<stem>")
def get_spectrogram(job_id, stem):
    if job_id not in JOBS:
        return jsonify({"error": "Unknown job"}), 404
    job = JOBS[job_id]
    if not job["spec_dir"]:
        return jsonify({"error": "Not ready"}), 202
    spec_path = Path(job["spec_dir"]) / f"{stem}.png"
    if not spec_path.exists():
        return jsonify({"error": "Stem not found"}), 404
    return send_file(str(spec_path), mimetype="image/png")


# ── Analysis worker ───────────────────────────────────────────────────────────

def _emit(job_id: str, stage: str, progress: int, message: str, **extra):
    job = JOBS[job_id]
    job["progress"] = progress
    job["message"]  = message
    event = {"stage": stage, "progress": progress, "message": message, **extra}
    job["queue"].put(event)
    print(f"  [{job_id}] {progress:3d}%  {message}")


def _run_analysis(job_id: str, audio_path: str):
    try:
        _emit(job_id, "load", 2, "Loading audio…")
        result = _analyze(job_id, audio_path)

        out_path = Path(audio_path).with_suffix(".analysis.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        JOBS[job_id]["result_path"] = str(out_path)
        JOBS[job_id]["status"]      = "done"
        _emit(job_id, "done", 100, "Analysis complete", result_job_id=job_id)

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"]  = str(e)
        _emit(job_id, "error", 0, f"Error: {e}")
        traceback.print_exc()


def _analyze(job_id: str, audio_path: str) -> dict:
    import librosa

    audio_path = str(audio_path)
    basename   = Path(audio_path).stem
    audio_dir  = Path(audio_path).parent
    stems_dir  = audio_dir / ".stems" / basename
    spec_dir   = stems_dir / "spectrograms"
    stems_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    JOBS[job_id]["spec_dir"] = str(spec_dir)

    SR  = 44100
    hop = 512

    # ── 1. Load ───────────────────────────────────────────────────────────────
    _emit(job_id, "load", 4, "Loading audio at 44100 Hz…")
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / sr
    _emit(job_id, "load", 8, f"Loaded {duration:.1f}s  |  {sr} Hz")

    # ── 2. Master analysis ────────────────────────────────────────────────────
    _emit(job_id, "master", 10, "Tracking beats (DBN)…")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames",
                                                  trim=False, tightness=100)
    beat_times     = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    downbeat_times = beat_times[::4]

    _emit(job_id, "master", 15, "Detecting master onsets…")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units="frames")
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    _emit(job_id, "master", 19, "Computing spectral features…")
    rms_frames    = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times_arr = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop)
    sc_frames     = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    _emit(job_id, "master", 22, "Detecting sections…")
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    feat   = np.vstack([librosa.util.normalize(mfcc,   axis=1),
                        librosa.util.normalize(chroma, axis=1)])
    n_segs = min(12, max(4, int(duration / 15)))
    try:
        bounds_frames = librosa.segment.agglomerative(feat, k=n_segs)
        bounds_times  = librosa.frames_to_time(bounds_frames, sr=sr).tolist()
    except Exception:
        bounds_times  = list(np.linspace(0, duration, n_segs + 1))

    sections = _label_sections(bounds_times, duration, rms_frames, rms_times_arr)

    _emit(job_id, "master", 26, "Estimating key…")
    chroma_mean = np.mean(chroma, axis=1)
    key_names   = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    key_str     = key_names[int(np.argmax(chroma_mean))]

    _emit(job_id, "master", 29, "Building master waveform & spectrogram…")
    master_peaks = _waveform_peaks(y, WAVEFORM_BINS)
    _save_spectrogram(y, sr, spec_dir / "master.png", "master")

    master_data = {
        "bpm":               float(np.round(float(tempo), 2)),
        "time_signature":    4,
        "key":               key_str,
        "beats":             [round(t, 4) for t in beat_times],
        "downbeats":         [round(t, 4) for t in downbeat_times],
        "onsets":            [round(t, 4) for t in onset_times],
        "sections":          sections,
        "rms":               rms_frames.tolist(),
        "rms_times":         rms_times_arr.tolist(),
        "spectral_centroid": sc_frames.tolist(),
        "sc_times":          rms_times_arr.tolist(),
    }

    # ── 3. Stem separation (demucs htdemucs_6s) ───────────────────────────────
    stem_paths = _separate_stems(job_id, audio_path, stems_dir)

    # ── 4. Per-stem analysis ──────────────────────────────────────────────────
    stems_data  = {}
    waveforms   = {"master": master_peaks}
    spec_urls   = {"master": f"/api/spectrogram/{job_id}/master"}
    loaded_audio = {}  # keep drums + other in memory for derived stems

    for i, stem_name in enumerate(STEMS):
        prog = 68 + int(i / len(STEMS) * 12)   # 68–80%
        _emit(job_id, "stems_analysis", prog, f"Analysing {stem_name}…")

        stem_path = stem_paths.get(stem_name)
        if stem_path and Path(stem_path).exists():
            ys, _ = librosa.load(str(stem_path), sr=SR, mono=True)
        else:
            ys = np.zeros_like(y)

        if stem_name in ("drums", "other"):
            loaded_audio[stem_name] = ys   # needed for derived stems

        stems_data[stem_name] = _analyse_stem(ys, SR, hop, stem_name)
        waveforms[stem_name]  = _waveform_peaks(ys, WAVEFORM_BINS)

        spec_out = spec_dir / f"{stem_name}.png"
        _save_spectrogram(ys, SR, spec_out, stem_name)
        spec_urls[stem_name] = f"/api/spectrogram/{job_id}/{stem_name}"

    # ── 5. Derived stems ──────────────────────────────────────────────────────
    _emit(job_id, "derived", 80, "HPSS split on drums (kick/snare vs cymbals)…")
    drums_y = loaded_audio.get("drums", np.zeros_like(y))
    other_y = loaded_audio.get("other", np.zeros_like(y))

    drums_harm, drums_perc = _hpss(drums_y)

    _emit(job_id, "derived", 83, "HPSS split on other (bells vs strings/choir)…")
    # For 'other' we still run the second-pass demucs for better separation,
    # but use HPSS as a fast fallback if demucs fails.
    second_pass = _second_pass_demucs(job_id, stem_paths.get("other", ""), stems_dir)

    _emit(job_id, "derived", 90, "Sub-bass frequency band extraction…")
    band_sub = _lowpass(y, SR, 80)

    derived_audio = {
        "drums_perc":    drums_perc,
        "drums_harm":    drums_harm,
        "other_bells":   second_pass.get("other_bells",   _hpss(other_y)[1]),  # perc fallback
        "other_choir":   second_pass.get("other_choir",   np.zeros_like(y)),
        "other_strings": second_pass.get("other_strings", _hpss(other_y)[0]),  # harm fallback
        "band_sub":      band_sub,
    }

    # Save HPSS-derived + sub-bass stems as WAVs so they can be played back
    # (2nd-pass demucs already saves other_bells/choir/strings above)
    _emit(job_id, "derived", 91, "Saving derived stem WAVs for playback…")
    for _dstem, _ddata in [("drums_perc", drums_perc),
                            ("drums_harm", drums_harm),
                            ("band_sub",   band_sub)]:
        _wav = stems_dir / f"{_dstem}.wav"
        if not _wav.exists():
            sf.write(str(_wav), _ddata, SR)

    _emit(job_id, "derived", 92, "Analysing derived stems…")
    for i, (stem_name, ys) in enumerate(derived_audio.items()):
        prog = 92 + int(i / len(derived_audio) * 5)
        _emit(job_id, "derived", prog, f"Analysing {stem_name}…")
        stems_data[stem_name] = _analyse_stem(ys, SR, hop, stem_name)
        waveforms[stem_name]  = _waveform_peaks(ys, WAVEFORM_BINS)
        spec_out = spec_dir / f"{stem_name}.png"
        _save_spectrogram(ys, SR, spec_out, stem_name)
        spec_urls[stem_name] = f"/api/spectrogram/{job_id}/{stem_name}"

    # ── 6. Assemble result ────────────────────────────────────────────────────
    _emit(job_id, "writing", 97, "Assembling JSON…")
    # Include all stems (primary + derived) that have WAV files on disk
    stem_audio_paths = {s: str(stems_dir / f"{s}.wav") for s in ALL_STEMS
                        if (stems_dir / f"{s}.wav").exists()}
    return {
        "version":          4,
        "source_file":      Path(audio_path).name,
        "duration":         round(duration, 3),
        "sample_rate":      SR,
        "master":           master_data,
        "stems":            stems_data,
        "waveforms":        waveforms,
        "spectrogram_urls": spec_urls,
        "stem_audio_paths": stem_audio_paths,
    }


# ── Per-stem analysis ─────────────────────────────────────────────────────────

def _analyse_stem(ys: np.ndarray, sr: int, hop: int, stem_name: str) -> dict:
    """Compute onsets, RMS, and spectral centroid for one stem."""
    import librosa
    delta = ONSET_DELTA.get(stem_name, 0.07)

    onset_env    = librosa.onset.onset_strength(y=ys, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,
                                               backtrack=True, delta=delta,
                                               units="frames")
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop).tolist()

    rms    = librosa.feature.rms(y=ys, hop_length=hop)[0]
    times  = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # Onset strength envelope (normalised) — useful for visualising attack intensity
    onset_env_norm = (onset_env / (onset_env.max() + 1e-9)).tolist()

    return {
        "onsets":        [round(t, 4) for t in onset_times],
        "rms":           rms.tolist(),
        "rms_times":     times.tolist(),
        "onset_env":     onset_env_norm,       # per-frame attack strength
    }


# ── HPSS split ────────────────────────────────────────────────────────────────

def _hpss(y: np.ndarray, margin: tuple = (1.0, 5.0)):
    """Harmonic-percussive source separation.
    Returns (harmonic, percussive).  margin=(h,p): higher p = sharper percussive."""
    import librosa
    return librosa.effects.hpss(y, margin=margin)


# ── Second-pass demucs on 'other' stem ───────────────────────────────────────

def _second_pass_demucs(job_id: str, other_wav: str, stems_dir: Path) -> dict:
    """Run htdemucs (4-stem) on the 'other' stem to reveal sub-components.

    Returns dict with keys: other_bells, other_choir, other_strings.
    Results are cached — skips if files already exist.
    """
    import torch, subprocess, sys

    # Cache paths
    cache = {
        "other_bells":   stems_dir / "other_bells.wav",
        "other_choir":   stems_dir / "other_choir.wav",
        "other_strings": stems_dir / "other_strings.wav",
    }
    import librosa

    if all(p.exists() for p in cache.values()):
        _emit(job_id, "derived", 84, "Using cached 2nd-pass stems…")
        result = {}
        for k, p in cache.items():
            ys, _ = librosa.load(str(p), sr=44100, mono=True)
            result[k] = ys
        return result

    if not other_wav or not Path(other_wav).exists():
        _emit(job_id, "derived", 84, "Skipping 2nd-pass (other.wav not found)…")
        return {}

    _emit(job_id, "derived", 84, "2nd-pass demucs on 'other' stem (htdemucs 4-stem)…")

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    tmp_out = stems_dir / "_demucs_2nd"
    tmp_out.mkdir(exist_ok=True)
    basename = Path(other_wav).stem   # "other"

    cmd = [
        sys.executable, "-m", "demucs",
        "--name", "htdemucs",
        "--device", device,
        "--out", str(tmp_out),
        other_wav,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        _emit(job_id, "derived", 87, f"2nd-pass demucs failed — skipping: {proc.stderr[:120]}")
        return {}

    demucs_out = tmp_out / "htdemucs" / basename
    # Map demucs output branches to semantic names for Christmas music:
    #   drums  → other_bells  (percussion in "other" = bells, glockenspiel, triangle)
    #   vocals → other_choir  (vocals in "other"    = choir, backing harmonies)
    #   other  → other_strings(other in "other"     = strings, pads, brass, synths)
    #   bass   → (discarded or merged into other_strings)
    branch_map = {
        "drums":  "other_bells",
        "vocals": "other_choir",
        "other":  "other_strings",
    }

    result = {}
    for branch, key in branch_map.items():
        src = demucs_out / f"{branch}.wav"
        dst = cache[key]
        if src.exists():
            src.rename(dst)
            ys, _ = librosa.load(str(dst), sr=44100, mono=True)
            result[key] = ys
        else:
            result[key] = np.zeros(44100, dtype=np.float32)

    _emit(job_id, "derived", 89, "2nd-pass demucs complete.")
    return result


# ── Primary stem separation ───────────────────────────────────────────────────

def _separate_stems(job_id: str, audio_path: str, stems_dir: Path) -> dict[str, str]:
    """Run demucs htdemucs_6s. Returns {stem_name: wav_path}. Skips if cached."""
    import torch

    basename = Path(audio_path).stem
    expected = {s: stems_dir / f"{s}.wav" for s in STEMS}

    if all(p.exists() for p in expected.values()):
        _emit(job_id, "stems", 42, "Using cached primary stems…")
        return {s: str(p) for s, p in expected.items()}

    _emit(job_id, "stems", 42, "Separating stems — demucs htdemucs_6s…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _emit(job_id, "stems", 44, f"  device: {device}")

    tmp_out = stems_dir / "_demucs_tmp"
    tmp_out.mkdir(exist_ok=True)

    import subprocess, sys
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", "htdemucs_6s",
        "--device", device,
        "--out", str(tmp_out),
        audio_path,
    ]
    _emit(job_id, "stems", 45, "  Running demucs htdemucs_6s…")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"demucs failed:\n{proc.stderr}")

    demucs_out = tmp_out / "htdemucs_6s" / basename
    result = {}
    for stem_name in STEMS:
        src = demucs_out / f"{stem_name}.wav"
        dst = expected[stem_name]
        if src.exists():
            src.rename(dst)
            result[stem_name] = str(dst)
        else:
            result[stem_name] = ""

    _emit(job_id, "stems", 68, "Primary stem separation complete.")
    return result


# ── Signal processing helpers ─────────────────────────────────────────────────

def _lowpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """4th-order Butterworth low-pass filter."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, cutoff / (sr / 2), btype="low", output="sos")
    return sosfilt(sos, y).astype(np.float32)


def _highpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(4, cutoff / (sr / 2), btype="high", output="sos")
    return sosfilt(sos, y).astype(np.float32)


def _bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    nyq = sr / 2
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfilt(sos, y).astype(np.float32)


# ── Waveform peaks ────────────────────────────────────────────────────────────

def _waveform_peaks(y: np.ndarray, n_bins: int) -> dict:
    n = len(y)
    if n == 0:
        return {"peaks_min": [0.0]*n_bins, "peaks_max": [0.0]*n_bins, "bins": n_bins}
    chunk = max(1, n // n_bins)
    trim  = (n // chunk) * chunk
    arr   = y[:trim].reshape(-1, chunk)
    mins  = arr.min(axis=1)
    maxs  = arr.max(axis=1)

    def _pad(a):
        if len(a) < n_bins:
            a = np.concatenate([a, np.zeros(n_bins - len(a))])
        return a[:n_bins].tolist()

    return {"peaks_min": _pad(mins), "peaks_max": _pad(maxs), "bins": n_bins}


# ── Spectrogram PNG ───────────────────────────────────────────────────────────

def _save_spectrogram(y: np.ndarray, sr: int, out_path: Path, title: str):
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hop   = max(1, len(y) // SPEC_TIME_BINS)
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=SPEC_MEL_BINS,
                                            hop_length=hop, fmax=8000)
    S_db  = librosa.power_to_db(S_mel, ref=np.max)

    color = STEM_COLORS.get(title, "#888888").lstrip("#")
    r, g, b = int(color[0:2],16)/255, int(color[2:4],16)/255, int(color[4:6],16)/255
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "stem", [(0,0,0), (r*0.3,g*0.3,b*0.3), (r,g,b), (1,1,1)], N=256
    )

    fig, ax = plt.subplots(figsize=(20, 2), dpi=100)
    ax.imshow(S_db, origin="lower", aspect="auto", cmap=cmap,
              vmin=-80, vmax=0, interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight", pad_inches=0,
                facecolor="black")
    plt.close(fig)


# ── Section labelling ─────────────────────────────────────────────────────────

def _label_sections(bounds: list, duration: float,
                    rms_frames: np.ndarray, rms_times: np.ndarray) -> list:
    if not bounds:
        return [{"start": 0.0, "end": round(duration, 3), "label": "main", "energy": 0.5}]

    bounds = sorted(set([0.0] + bounds + [duration]))
    segs   = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i+1]
        mask = (rms_times >= s) & (rms_times < e)
        energy = float(np.mean(rms_frames[mask])) if mask.any() else 0.0
        segs.append({"start": round(s, 3), "end": round(e, 3),
                     "label": "", "energy": round(energy, 4)})

    if not segs:
        return segs

    energies = [s["energy"] for s in segs]
    e_max    = max(energies) or 1.0

    n = len(segs)
    for i, seg in enumerate(segs):
        ratio = seg["energy"] / e_max
        if i == 0:
            label = "intro"
        elif i == n - 1:
            label = "outro"
        elif ratio >= 0.80:
            label = "chorus"
        elif ratio <= 0.45:
            label = "bridge"
        else:
            label = "verse"
        seg["label"] = label

    return segs


# ── Sequence extension (xsq_extend) ───────────────────────────────────────────
# Extends a partly hand-sequenced .xsq by repeating the sequenced material wherever
# the song repeats. The engine lives in xsq_extend.py; these endpoints wrap it with
# caching, versioned output and the promote step. The original file is never written:
# every apply produces a new numbered sibling (see _next_version).

import glob as _glob
import hashlib
import xsq_extend

VERSION_RE = re.compile(r"^(?P<base>.+?)\.(?P<major>\d+)\.(?P<minor>\d+)$")
VERSION_ARCHIVE = ".pc_versions"


def _split_version(p: Path):
    """'back-yard-master.0.3.xsq' -> ('back-yard-master', (0, 3)).
    An unversioned file is the base, implicitly version 0.0 -> (stem, None)."""
    m = VERSION_RE.fullmatch(p.stem)
    if not m:
        return p.stem, None
    return m.group("base"), (int(m.group("major")), int(m.group("minor")))


def _versions_of(p: Path):
    """Every version sibling sharing p's base stem, sorted NUMERICALLY (0.10 after 0.2)."""
    base, _ = _split_version(p)
    out = []
    for f in p.parent.glob(_glob.escape(base) + ".*.xsq"):
        b, v = _split_version(f)
        if b == base and v:
            out.append((v, f))
    return sorted(out)


def _next_version(p: Path) -> Path:
    """Next free version name for p's base stem. Strips any version already on p, so
    applying to ...0.3.xsq gives ...0.4.xsq, never ...0.3.0.1.xsq."""
    base, _ = _split_version(p)
    vs = _versions_of(p)
    major, minor = vs[-1][0] if vs else (0, 0)
    return p.parent / f"{base}.{major}.{minor + 1}.xsq"


def _sidecar(p: Path, kind: str) -> Path:
    """kind: 'extend-plan.json' | 'added.json' — sits beside the sequence it describes."""
    return p.parent / f"{p.stem}.{kind}"


def _xsq_sort_key(p: Path):
    base, v = _split_version(p)
    return (base.lower(), v or (0, 0))


def _model_groups() -> set:
    """xLights model GROUP names. PixelConductor writes groups as <Element type="model">
    with an empty EffectLayer, so without this the engine can treat one as a model and
    'extend' mode writes effects onto a group."""
    import xml.etree.ElementTree as ET
    xml_path = XLIGHTS_SHOW_DIR / "xlights_rgbeffects.xml"
    if not xml_path.exists():
        return set()
    try:
        root = ET.parse(str(xml_path)).getroot()
        return {g.get("name", "") for g in root.findall(".//modelGroups/modelGroup")}
    except Exception:
        return set()


def _count_effects(p: Path) -> int:
    """Timeline effects on models: not EffectDB entries, not timing-track marks."""
    import xml.etree.ElementTree as ET
    try:
        ee = ET.parse(str(p)).getroot().find("ElementEffects")
    except Exception:
        return 0
    if ee is None:
        return 0
    n = 0
    for el in ee:
        if el.get("type") != "model":
            continue
        for lay in el.findall("EffectLayer"):
            n += len(lay)
        for sl in el.findall("SubModelEffectLayer"):
            n += len(sl)
    return n


def _app_beats(audio: Path):
    """The beat grid PixelConductor's own analysis cached, so the extend plan lines up
    with the beat lines the editor already draws. None when the song isn't analysed."""
    aj = audio.with_suffix(".analysis.json")
    if not aj.exists():
        return None
    try:
        with open(aj, encoding="utf-8") as f:
            return json.load(f)["master"]["beats"] or None
    except Exception:
        return None


def _extend_analysis(audio: Path, beats=None) -> dict:
    """analyze() is ~20 s and its similarity matrix is far too big to ship to the browser
    or store as JSON, so it is cached per song as <audio>.extend.npz. The cache records
    which beat grid produced it and is recomputed when that changes."""
    sig = "librosa" if beats is None else hashlib.sha1(
        ",".join(f"{float(b):.4f}" for b in beats).encode()).hexdigest()[:12]
    cache = audio.with_suffix(".extend.npz")
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["sig"]) == sig:
                a = {k: z[k] for k in z.files if k != "sig"}
                a["tempo"] = float(a["tempo"])
                a["phase"] = int(a["phase"])
                a["duration"] = float(a["duration"])
                a["beat_source"] = "supplied" if beats is not None else "librosa"
                return a
        except Exception:
            pass
    a = xsq_extend.analyze(str(audio), beats=beats)
    try:
        np.savez_compressed(cache, sig=np.array(sig),
                            **{k: v for k, v in a.items() if k != "beat_source"})
    except Exception as e:
        print(f"  extend: could not cache analysis: {e}")
    return a


def _coverage(added: list, duration: float, bins: int = 1200) -> list:
    """Added-effect density downsampled to `bins`, for the review strip's preview lane."""
    arr = [0] * bins
    if duration <= 0:
        return arr
    for _m, _sub, s, e, _tag in added:
        x0 = int(s / 1000.0 / duration * bins)
        x1 = int(e / 1000.0 / duration * bins) + 1
        for x in range(max(0, x0), min(bins, max(x1, x0 + 1))):
            arr[x] += 1
    return arr


def _extend_inputs(data: dict):
    """Shared validation. Returns (xsq_path, audio_path, analysis); raises ValueError."""
    xsq = Path(data.get("xsq", ""))
    audio = Path(data.get("audio", ""))
    if not xsq.exists() or xsq.suffix.lower() != ".xsq":
        raise ValueError(f"Sequence not found: {xsq}")
    if not audio.exists():
        raise ValueError(f"Audio not found: {audio}")
    # Default to the engine's own beat tracking: it is what the prototype was verified
    # on, and on "Wizards" app.py's tracker reports 147.67 BPM against the true 152,
    # which drags every lag_beats mapping off. use_app_beats trades that for a grid
    # identical to the one the editor draws.
    beats = _app_beats(audio) if data.get("use_app_beats") else None
    return xsq, audio, _extend_analysis(audio, beats)


@app.route("/api/extend/analyze", methods=["POST"])
def extend_analyze():
    data = request.get_json() or {}
    try:
        xsq, audio, a = _extend_inputs(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    try:
        plan = xsq_extend.propose_plan(a, str(xsq), data.get("cutoff_s"))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"propose_plan failed: {e}"}), 500
    saved_plan = None
    saved = _sidecar(xsq, "extend-plan.json")
    if saved.exists():
        try:
            saved_plan = json.loads(saved.read_text(encoding="utf-8"))
        except Exception:
            pass
    return jsonify({
        "plan":         plan,
        "saved_plan":   saved_plan,      # a previously edited plan for this same file
        "beats":        [round(float(b), 4) for b in a["beats"]],
        "beat_source":  a.get("beat_source", "librosa"),
        "tempo":        round(a["tempo"], 2),
        "duration":     round(a["duration"], 3),
        "phase":        a["phase"],
        "next_version": _next_version(xsq).name,
    })


@app.route("/api/extend/apply", methods=["POST"])
def extend_apply():
    data = request.get_json() or {}
    plan = data.get("plan") or {}
    if not plan.get("sections"):
        return jsonify({"error": "Plan has no sections"}), 400
    try:
        xsq, audio, a = _extend_inputs(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    dry = bool(data.get("dry_run"))
    try:
        r = xsq_extend.apply_plan(str(xsq), a, plan, groups=_model_groups(), dry_run=dry)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"apply_plan failed: {e}"}), 500

    if dry:
        return jsonify({"ok": True, "dry_run": True, "stats": r["stats"],
                        "added": len(r["added"]),
                        "coverage": _coverage(r["added"], a["duration"])})

    # Versioned output — never the input path, never an existing file.
    name = (data.get("out_name") or "").strip() or _next_version(xsq).name
    if not name.lower().endswith(".xsq"):
        name += ".xsq"
    if re.search(r'[\\/:*?"<>|]', name):
        return jsonify({"error": "A file name can't contain \\ / : * ? \" < > |"}), 400
    if _split_version(Path(name))[1] is None:
        return jsonify({"error": f"'{name}' is not a version name (expected <base>.<n>.<n>.xsq)"}), 400
    out = xsq.parent / name
    if out.resolve() == xsq.resolve():
        return jsonify({"error": "Refusing to overwrite the sequence being extended"}), 400
    if out.exists():
        return jsonify({"error": f"{name} already exists"}), 400

    plan_out = dict(plan)
    plan_out["from"] = xsq.name
    plan_out["beat_source"] = a.get("beat_source", "librosa")
    with SAVE_LOCK:
        try:
            _atomic_write(out, r["xml"])
            _atomic_write(_sidecar(out, "added.json"), json.dumps(r["added"]))
            _atomic_write(_sidecar(out, "extend-plan.json"), json.dumps(plan_out, indent=1))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "out_path": str(out), "out_name": out.name,
                    "stats": r["stats"], "added": len(r["added"]),
                    "effects": _count_effects(out)})


@app.route("/api/extend/plan", methods=["GET"])
def get_extend_plan():
    f = _sidecar(Path(request.args.get("xsq", "")), "extend-plan.json")
    if not f.exists():
        return jsonify({"error": "No saved plan"}), 404
    return Response(f.read_text(encoding="utf-8"), mimetype="application/json")


@app.route("/api/extend/plan", methods=["POST"])
def put_extend_plan():
    data = request.get_json() or {}
    xsq = Path(data.get("xsq", ""))
    if not xsq.parent.exists():
        return jsonify({"error": f"Folder not found: {xsq.parent}"}), 400
    with SAVE_LOCK:
        _atomic_write(_sidecar(xsq, "extend-plan.json"),
                      json.dumps(data.get("plan") or {}, indent=1))
    return jsonify({"ok": True})


@app.route("/api/extend/added")
def extend_added():
    """Provenance manifest for a sequence: which effects Extend generated, so the editor
    can tell them from hand-placed ones. 404 simply means 'nothing generated here'."""
    f = _sidecar(Path(request.args.get("xsq", "")), "added.json")
    if not f.exists():
        return jsonify({"error": "No manifest"}), 404
    return Response(f.read_text(encoding="utf-8"), mimetype="application/json")


@app.route("/api/extend/versions")
def extend_versions():
    """The whole chain for a base stem — the base file plus every numbered version."""
    xsq = Path(request.args.get("xsq", ""))
    if not xsq.parent.exists():
        return jsonify({"error": "Folder not found"}), 400
    base, _ = _split_version(xsq)
    rows = []
    for ver, f in [(None, xsq.parent / f"{base}.xsq")] + _versions_of(xsq):
        if not f.exists():
            continue
        n_added, src = None, None
        added = _sidecar(f, "added.json")
        if added.exists():
            try:
                n_added = len(json.loads(added.read_text(encoding="utf-8")))
            except Exception:
                pass
        plan_f = _sidecar(f, "extend-plan.json")
        if plan_f.exists():
            try:
                pj = json.loads(plan_f.read_text(encoding="utf-8"))
                src = pj.get("from") or pj.get("promoted_from")
            except Exception:
                pass
        rows.append({"name": f.name, "path": str(f),
                     "version": f"{ver[0]}.{ver[1]}" if ver else "base",
                     "mtime": f.stat().st_mtime, "size": f.stat().st_size,
                     "effects": _count_effects(f), "added": n_added, "from": src})
    return jsonify({"base": f"{base}.xsq", "versions": rows})


@app.route("/api/extend/promote", methods=["POST"])
def extend_promote():
    """Make a version the master: back the base up, overwrite it, archive the rest.

    Nothing is deleted. Superseded versions move to .pc_versions/ the way discarded
    drafts move to drafts/discarded/ — they are work that was never merged. The promoted
    version's own file is the exception: its bytes are now the base, so it is redundant.
    """
    data = request.get_json() or {}
    ver_path = Path(data.get("version", ""))
    if not ver_path.exists() or ver_path.suffix.lower() != ".xsq":
        return jsonify({"error": f"Version not found: {ver_path}"}), 400
    base, v = _split_version(ver_path)
    if v is None:
        return jsonify({"error": f"{ver_path.name} is the base, not a version"}), 400
    base_path = ver_path.parent / f"{base}.xsq"
    content = ver_path.read_text(encoding="utf-8")

    archived = []
    with SAVE_LOCK:
        # 1. Never overwrite a base we could not back up first.
        backup = None
        if base_path.exists():
            try:
                _backup_xsq(base_path)
                baks = sorted((base_path.parent / ".pc_backups").glob(f"{base}_*.xsq"))
                backup = baks[-1].name if baks else None
            except Exception as e:
                return jsonify({"error": f"Backup failed, nothing written: {e}"}), 500
        # 2. The version becomes the base.
        try:
            _atomic_write(base_path, content)
        except Exception as e:
            return jsonify({"error": f"Could not write {base_path.name}: {e}"}), 500
        # 3. Its sidecars move onto the base name, so the master records its provenance.
        try:
            src_plan = _sidecar(ver_path, "extend-plan.json")
            pj = json.loads(src_plan.read_text(encoding="utf-8")) if src_plan.exists() else {}
            pj["promoted_from"] = ver_path.name
            _atomic_write(_sidecar(base_path, "extend-plan.json"), json.dumps(pj, indent=1))
            src_added = _sidecar(ver_path, "added.json")
            if src_added.exists():
                _atomic_write(_sidecar(base_path, "added.json"),
                              src_added.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  promote: sidecar copy failed: {e}")
        # 4. Archive everything else; the promoted file itself is now redundant.
        if not data.get("keep_versions"):
            arch = base_path.parent / VERSION_ARCHIVE / f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"
            for _vv, f in _versions_of(ver_path):
                for g in (f, _sidecar(f, "added.json"), _sidecar(f, "extend-plan.json")):
                    if not g.exists():
                        continue
                    try:
                        if f == ver_path:
                            g.unlink()          # content survives as the base
                        else:
                            arch.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(g), str(arch / g.name))
                            archived.append(g.name)
                    except OSError as e:
                        print(f"  promote: could not archive {g.name}: {e}")
    return jsonify({"ok": True, "base_path": str(base_path), "base_name": base_path.name,
                    "backup": backup, "archived": archived,
                    "effects": _count_effects(base_path)})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("PixelConductor server starting…")
    print(f"  Open:  http://localhost:{PORT}")
    print(f"  HTML:  {HTML_FILE}")
    print(f"  Audio: {XLIGHTS_AUDIO}")
    print()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
