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
import time

import numpy as np
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline
import integrations

integrations.init_sentry()

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
    if crater_id not in SCENES:
        raise HTTPException(404, "unknown scene id")
    d = SCENES[crater_id]["dir"]
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
    for cid, s in SCENES.items():
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
    payload["scene_name"] = SCENES[crater_id]["name"]
    payload["source_image"] = "/static/%s/source.png" % SCENES[crater_id]["slug"]
    payload["reference_image"] = "/static/%s/reference.png" % SCENES[crater_id]["slug"]
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


@app.post("/analyze_upload")
async def analyze_upload(
    file: UploadFile = File(...),
    sun_az: float = Form(315.0),
    sun_el: float = Form(30.0),
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
    img = np.asarray(img.resize((512, 512), PILImage.LANCZOS),
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
    return {
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
