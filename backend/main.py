"""SIH26166 - FastAPI backend.

Endpoints (per the Tomorrow Plan):
  /health               -> liveness + learned-model flag
  /craters              -> limited demo DB (real OHRC scene + Tycho stand-in)
  /match/{crater_id}    -> run Stages 3-6, return percentage match payload
  /terrain/{crater_id}  -> Stage 7, Fourier-smoothed height grid + contours
Static: /  serves frontend/index.html, /static serves processed imagery.
"""
import json
import os
import re
import time

import numpy as np
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline
import ingest
import integrations

integrations.init_sentry()

# dynamic scenes registered via /ingest_product or /analyze_upload
_registry = ingest.load_registry()


def _all_scenes():
    merged = dict(SCENES)
    merged.update(_registry)
    return merged

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
FRONTEND = os.path.join(ROOT, "frontend")

SCENES = {
    "ohrc_20210401": {
        "dir": os.path.join(PROC, "ohrc_real"),
        "slug": "ohrc_real",
        "name": "CH-2 OHRC real scene",
        "subtitle": "ch2_ohr_ncp_20210401T2357376656 - 13.47S, 25.19E",
        "kind": "real",
    },
    "tycho": {
        "dir": os.path.join(PROC, "tycho_synthetic"),
        "slug": "tycho_synthetic",
        "name": "Tycho (synthetic stand-in)",
        "subtitle": "43.37S, 348.68E - synthetic DEM pair",
        "kind": "synthetic",
    },
}

GRID_N = 192          # height grid sent to the 3D mesh
CONTOUR_LEVELS = 8

# ensure static dirs exist even on fresh clones (data/ is gitignored)
os.makedirs(PROC, exist_ok=True)

app = FastAPI(title="SIH26166 backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])

_cache = {"match": {}, "scene_meta": {}}


def _scene_dir(crater_id):
    scenes = _all_scenes()
    if crater_id not in scenes:
        raise HTTPException(404, "unknown scene id")
    d = scenes[crater_id]["dir"]
    if not os.path.isdir(d):
        raise HTTPException(503, "scene not preprocessed yet - run "
                                 "backend/preprocess.py first")
    return d


def _load_pair(d):
    src = cv2.imread(os.path.join(d, "source.png"), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(os.path.join(d, "reference.png"), cv2.IMREAD_GRAYSCALE)
    dem = np.load(os.path.join(d, "dem.npy"))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    return src, ref, dem, meta


@app.get("/health")
def health():
    import pipeline
    loaded = pipeline.learned_model_loaded()
    return {"status": "ok", "learned_model_loaded": loaded,
            "pipeline": ("SIFT + learned ONNX descriptor" if loaded
                         else "SIFT fallback (descriptor.onnx not trained yet)")}


@app.get("/craters")
def craters():
    out = []
    for cid, s in _all_scenes().items():
        entry = {"id": cid, "name": s["name"], "subtitle": s["subtitle"],
                 "kind": s["kind"]}
        d = s["dir"]
        if os.path.isdir(d):
            meta = json.load(open(os.path.join(d, "metadata.json")))
            entry["metadata"] = meta
        out.append(entry)
    return out


@app.get("/match/{crater_id}")
def match(crater_id: str):
    # Cache-aside (Modern Web + AI Stack Guide §3.3):
    # 1. in-memory cache -> 2. Redis (TTL 300s) -> 3. fresh pipeline compute.
    # The fresh path is the only one that writes Supabase rows.
    if crater_id in _cache["match"]:
        payload = dict(_cache["match"][crater_id], cache="memory")
        return payload

    d = _scene_dir(crater_id)
    cached = integrations.redis_get_json("match:%s" % crater_id)
    if cached:
        _cache["match"][crater_id] = cached
        return dict(cached, cache="redis")

    src, ref, dem, meta = _load_pair(d)
    integrations.redis_set("job:%s:status" % crater_id,
                           json.dumps({"stage": "matching",
                                       "status": "running"}), ttl_seconds=600)
    integrations.breadcrumb("match start", data={"scene": crater_id})
    payload = pipeline.match_pair(src, ref)
    payload["scene_id"] = crater_id
    payload["scene_name"] = _all_scenes()[crater_id]["name"]
    payload["source_image"] = "/static/%s/source.png" % _all_scenes()[crater_id]["slug"]
    payload["reference_image"] = "/static/%s/reference.png" % _all_scenes()[crater_id]["slug"]
    payload["metadata"] = meta
    payload["evaluation_notes"] = (
        "RMSE reported in reference-image pixels. When RMSE approaches the "
        "image's own local-gradient information ceiling, that is the "
        "Cramer-Rao bound (Var >= sigma^2 / sum (dI/dx)^2) - a sharper rim "
        "gives a tighter bound; flat regolith makes lower RMSE impossible.")
    payload["cache"] = "fresh"
    craters_path = os.path.join(d, "craters.json")
    if os.path.exists(craters_path):
        with open(craters_path) as f:
            payload["craters"] = json.load(f)
    _cache["match"][crater_id] = payload

    # persistence: jobs -> matches -> metrics (scenes upserted separately)
    integrations.redis_set("job:%s:status" % crater_id,
                           json.dumps({"stage": "done",
                                       "status": "ok",
                                       "match_percentage":
                                           payload.get("match_percentage")}),
                           ttl_seconds=600)
    integrations.redis_set_json("match:%s" % crater_id, payload, ttl_seconds=1800)
    integrations.supabase_upsert_scene({
        "product_id": meta.get("product_id"),
        "instrument": meta.get("instrument"),
        "source_scene": crater_id,
        "footprint": meta.get("footprint_corners"),
        "metadata": meta,
    })
    integrations.supabase_insert("jobs", {"stage": "match", "status": "done",
                                          "source_scene": crater_id})
    integrations.supabase_insert("matches", {
        "source_scene": crater_id,
        "keypoints_source": payload.get("keypoints_source", [])[:500],
        "keypoints_ref": payload.get("keypoints_ref", [])[:500],
        "homography": payload.get("homography"),
        "match_percentage": payload.get("match_percentage"),
    })
    integrations.supabase_insert("metrics", {
        "source_scene": crater_id,
        "rmse": payload.get("rmse_px"),
        "inlier_count": payload.get("inlier_count"),
        "inlier_ratio": payload.get("inlier_ratio"),
        "match_percentage": payload.get("match_percentage"),
        "method": ("sift+learned"
                   if payload.get("method_breakdown", {}).get("learned_model_loaded")
                   else "sift"),
        "ransac_k_derived": payload.get("ransac", {}).get("derived_iterations_k"),
    })
    integrations.breadcrumb("match complete", data={
        "scene": crater_id, "match_pct": payload.get("match_percentage")})
    return payload


@app.get("/terrain/{crater_id}")
def terrain(crater_id: str):
    # cache-aside level 2: Redis (payload is rounded to fit Upstash caps)
    cached = integrations.redis_get_json("terrain:%s" % crater_id)
    if cached:
        return dict(cached, cache="redis")

    d = _scene_dir(crater_id)
    _, _, dem, meta = _load_pair(d)

    # Problem 7 - truncated Fourier low-pass, then re-tessellate
    smooth = pipeline.fourier_smooth(dem, keep_fraction=0.18)
    g = cv2.resize(smooth, (GRID_N, GRID_N),
                   interpolation=cv2.INTER_AREA).astype(float)
    g -= g.min()
    cell_m = meta.get("analysis_grid", {}).get(
        "cell_meters", 1.0) * (dem.shape[0] / GRID_N)

    zmin, zmax = float(g.min()), float(g.max())
    levels = [zmin + (zmax - zmin) * (i + 1) / (CONTOUR_LEVELS + 1)
              for i in range(CONTOUR_LEVELS)]
    contours = {"levels_m": levels,
                "segments": [pipeline.marching_squares(g, lv)
                             for lv in levels]}

    # selenographic footprint of this DEM patch (real scenes only)
    footprint = meta.get("footprint_corners")
    payload = {
        "scene_id": crater_id,
        "grid": {"n": GRID_N, "cell_meters": float(cell_m),
                 "extent_m": float(cell_m * GRID_N),
                 "heights_m": [[round(float(v), 3) for v in row] for row in g],
                 "zmin_m": zmin, "zmax_m": zmax},
        "contours": contours,
        "metadata": meta,
        "cache": "fresh",
    }
    integrations.redis_set_json("terrain:%s" % crater_id, payload,
                                ttl_seconds=600)
    return payload


@app.get("/narrate/{crater_id}")
def narrate(crater_id: str):
    """Judge-facing narration of the metrics - Gemini narrates, it NEVER
    generates the metrics (per the implementation plan). Rate-limited via
    Upstash Redis (NARRATE_RATE_LIMIT/min) when configured."""
    allowed, n = integrations.redis_rate_limit(
        "narrate:%s" % crater_id, integrations.NARRATE_RATE_LIMIT, 60)
    if not allowed:
        raise HTTPException(429, "narration rate limit exceeded (%d/min)"
                            % integrations.NARRATE_RATE_LIMIT)
    m = match(crater_id)
    mb = m.get("method_breakdown", {})
    summary = (
        "Scene %s matched at %.1f%% (%d/%d inliers), RMSE %.2f px, "
        "SIFT candidates %d, learned branch %s (%d candidates), "
        "RANSAC w=%.2f, derived k=%d."
        % (crater_id, m.get("match_percentage", 0), m.get("inlier_count", 0),
           len(m.get("matches", [])), m.get("rmse_px") or 0,
           mb.get("sift_candidates", 0),
           "loaded" if mb.get("learned_model_loaded") else "not trained",
           mb.get("learned_candidates", 0),
           m.get("ransac", {}).get("inlier_fraction_w", 0),
           m.get("ransac", {}).get("derived_iterations_k", 0)))
    text = integrations.gemini_narrate(
        "You are narrating lunar image-registration results to a "
        "non-technical hackathon judge. Be precise, honest, under 120 words. "
        "Do not invent numbers beyond these: " + summary)
    source = "gemini" if text else "local-template"
    if not text:
        text = integrations.local_narration(m)

    # matches + metrics rows are written on the fresh-compute path of
    # /match (cache-aside). Here we only narrate.
    return {"narration": text, "source": source, "rate_used": n,
            "metrics_summary": summary}


@app.post("/ingest_product")
def ingest_product(path: str, scene_id: str = None,
                   crop_line: int = None, crop_sample: int = None,
                   crop_size: int = 4096,
                   sun_az: float = 270.8, sun_el: float = 10.0):
    """Turn ANY Chandrayaan-2 PDS4 product (XML+IMG) into a full matchable
    scene: SFS DEM, craters, terrain. `path` is a directory under data/raw
    (or an absolute path). crop_line/crop_sample pick any region; omit them
    to auto-select the most feature-rich window."""
    import ingest as ing
    p = path if os.path.isabs(path) else os.path.join(ROOT, "data", "raw", path)
    if not os.path.isdir(p):
        raise HTTPException(404, "product directory not found: %s" % p)
    crop = (crop_line, crop_sample) if crop_line is not None else None
    try:
        res = ing.ingest_product_dir(p, scene_id=scene_id, crop=crop,
                                     crop_size=crop_size, sun_az=sun_az,
                                     sun_el=sun_el)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    global _registry
    _registry = ingest.load_registry()
    integrations.breadcrumb("scene ingested", data={
        "scene": res["scene_id"], "product": p})
    return res


def _terrain_payload(scene_id, dem, cell_m, meta, cache="fresh"):
    """Shared Stage-7 payload: truncated-Fourier smoothed grid + marching
    squares contours - exactly what the 3D hologram consumes."""
    smooth = pipeline.fourier_smooth(dem, keep_fraction=0.18)
    g = cv2.resize(smooth, (GRID_N, GRID_N),
                   interpolation=cv2.INTER_AREA).astype(float)
    g -= g.min()
    zmin, zmax = float(g.min()), float(g.max())
    levels = [zmin + (zmax - zmin) * (i + 1) / (CONTOUR_LEVELS + 1)
              for i in range(CONTOUR_LEVELS)]
    return {
        "scene_id": scene_id,
        "grid": {"n": GRID_N, "cell_meters": float(cell_m),
                 "extent_m": float(GRID_N * cell_m),
                 "heights_m": [[round(float(v), 3) for v in row] for row in g],
                 "zmin_m": zmin, "zmax_m": zmax},
        "contours": {"levels_m": levels,
                     "segments": [pipeline.marching_squares(g, lv)
                                  for lv in levels]},
        "metadata": meta,
        "cache": cache,
    }


@app.post("/ingest_product_upload")
async def ingest_product_upload(
    files: list[UploadFile] = File(...),
    scene_name: str = Form(""),
    crop_size: int = Form(4096),
    sun_az: float = Form(270.8),
    sun_el: float = Form(10.0),
):
    """Upload a real mission product DIRECTLY - an ISRO PDS4 XML+IMG pair,
    a NASA PDS3 .XML+.IMG pair, or a ZIP of the product directory - and get
    a full matchable scene (SFS DEM, craters, auto-selected reference,
    terrain payload). Plain images use /analyze_upload instead."""
    import io
    import zipfile
    import tempfile
    if not files:
        raise HTTPException(400, "no files uploaded")
    names = [f.filename or "" for f in files]
    product_exts = (".zip", ".img", ".xml")
    if not any(n.lower().endswith(product_exts) for n in names):
        raise HTTPException(400, "expected a PDS4/PDS3 product (.xml + .img "
                                 "pair, or a .zip of the product directory)")
    tmp = tempfile.mkdtemp(prefix="sih_product_")
    try:
        if len(files) == 1 and names[0].lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(await files[0].read())) as z:
                z.extractall(tmp)
        else:
            for f in files:
                dest = os.path.join(tmp, os.path.basename(f.filename
                                                          or "unnamed"))
                with open(dest, "wb") as out:
                    out.write(await f.read())
        sid_base = scene_name or os.path.splitext(names[0])[0]
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", sid_base)[:48]
        sid = "up_prod_%s_%d" % (sid, int(time.time()) % 100000)
        res = ingest.ingest_product_dir(tmp, scene_id=sid,
                                        crop_size=crop_size, sun_az=sun_az,
                                        sun_el=sun_el)
        global _registry
        _registry = ingest.load_registry()
        d = res["entry"]["dir"]
        dem = np.load(os.path.join(d, "dem.npy"))
        meta = json.load(open(os.path.join(d, "metadata.json")))
        integrations.breadcrumb("product uploaded", data={
            "scene": res["scene_id"], "files": names})
        return {
            "created_scene": res["scene_id"],
            "terrain": _terrain_payload(res["scene_id"], dem,
                                        meta.get("analysis_grid", {})
                                        .get("cell_meters", 1.0), meta),
            "label": res.get("label", {}),
            "craters": res.get("craters", []),
        }
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:                                   # noqa: BLE001
        integrations.capture_exception(exc)
        raise HTTPException(500, "product ingestion failed: %s" % exc)


@app.post("/analyze_upload")
async def analyze_upload(
    file: UploadFile = File(...),
    sun_az: float = Form(315.0),
    sun_el: float = Form(30.0),
    make_scene: bool = Form(False),
    scene_name: str = Form(""),
):
    """Upload any lunar/surface image -> shape-from-shading relief -> the
    same terrain payload the 3D hologram consumes (grid + marching-squares
    contours). Non-metric, slope-calibrated relief - same honesty framing
    as the OHRC scene."""
    import io
    from PIL import Image as PILImage
    from preprocess import shape_from_shading

    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "file too large (25 MB max)")
    try:
        img = PILImage.open(io.BytesIO(data)).convert("L")
    except Exception:
        raise HTTPException(400, "could not decode image")
    img = np.asarray(img.resize((1024, 1024), PILImage.LANCZOS),
                     dtype=np.float32)

    dem = shape_from_shading(img, sun_az, sun_el, cell_m=1.0)
    smooth = pipeline.fourier_smooth(dem, keep_fraction=0.18)
    g = cv2.resize(smooth, (GRID_N, GRID_N),
                   interpolation=cv2.INTER_AREA).astype(float)
    g -= g.min()
    zmin, zmax = float(g.min()), float(g.max())
    levels = [zmin + (zmax - zmin) * (i + 1) / (CONTOUR_LEVELS + 1)
              for i in range(CONTOUR_LEVELS)]
    contours = {"levels_m": levels,
                "segments": [pipeline.marching_squares(g, lv)
                             for lv in levels]}
    result = {
        "scene_id": "upload:%s" % (file.filename or "image"),
        "grid": {"n": GRID_N, "cell_meters": 1.0,
                 "extent_m": float(GRID_N),
                 "heights_m": [[round(float(v), 3) for v in row] for row in g],
                 "zmin_m": zmin, "zmax_m": zmax},
        "contours": contours,
        "metadata": {
            "product_id": "user-upload",
            "instrument": "Uploaded image (shape-from-shading)",
            "sun": {"sun_azimuth_deg": sun_az, "sun_elevation_deg": sun_el},
            "provenance": "User-uploaded image; relief reconstructed by "
                          "linearized-Lambertian shape-from-shading - "
                          "photometric approximation, NOT a metric DEM.",
        },
        "cache": "fresh",
    }
    if make_scene:
        scene = _make_scene_from_upload(
            np.asarray(img, dtype=np.uint8), file.filename, sun_az, sun_el,
            scene_name)
        global _registry
        _registry = ingest.load_registry()
        result["created_scene"] = scene["scene_id"]
        result["craters"] = scene["craters"]
    return result


def _make_scene_from_upload(img_u8, filename, sun_az, sun_el, scene_name):
    """Promote an uploaded image to a full matchable scene (SFS DEM +
    craters) registered in the scene registry."""
    import ingest as ing
    sid = re.sub(r"[^A-Za-z0-9_-]", "_",
                 scene_name or ("upload_%s" % os.path.splitext(
                     filename or "image")[0]))[:48]
    sid = "up_%s_%d" % (sid, int(time.time()) % 100000)
    return ing.ingest_image(
        img_u8, sid, cell_m=1.0, sun_az=sun_az, sun_el=sun_el,
        provenance="Full scene created from an uploaded image; relief is a "
                   "photometric approximation (non-metric).",
        product_id=filename or "uploaded image")


@app.get("/debug/sentry-test")
def sentry_test():
    """Deliberate test error so you can watch it land in the Sentry
    dashboard (Issues -> sih26166-backend)."""
    try:
        raise RuntimeError("SIH26166 sentry verification event - safe to ignore")
    except RuntimeError as exc:
        integrations.capture_exception(exc)
        integrations.breadcrumb("sentry test event sent", "debug")
        return {"sent": True, "note": "check Sentry Issues -> sih26166-backend"}


app.mount("/static", StaticFiles(directory=PROC), name="static")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
