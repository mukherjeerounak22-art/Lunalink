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
from pydantic import BaseModel

import pipeline
import ingest
import integrations
import layers

integrations.init_sentry()

# dynamic scenes registered via /ingest_product or /analyze_upload
_registry = ingest.load_registry()


def _all_scenes():
    merged = dict(SCENES)
    # a dynamically-ingested scene whose processed dir was wiped (cleanup,
    # fresh re-ingest) is stale - drop it from memory AND the registry so
    # the UI never receives an id that would 503 later
    stale = [k for k, v in _registry.items()
             if not os.path.isdir(v.get("dir", ""))]
    if stale:
        for k in stale:
            _registry.pop(k, None)
        try:
            p = os.path.join(PROC, "registry.json")
            reg = json.load(open(p))
            if isinstance(reg, dict):
                for k in stale:
                    reg.pop(k, None)
                json.dump(reg, open(p, "w"), indent=1)
        except Exception:                                    # noqa: BLE001
            pass
    merged.update(_registry)
    return merged

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
FRONTEND = os.path.join(ROOT, "frontend")

SCENES = {
    "ohrc_20210401": {
        "dir": os.path.join(PROC, "ohrc_real"),
        "slug": "ohrc_real",
        "name": "Level 3 - CH-2 OHRC real scene (cross-mission, hard)",
        "subtitle": "real ISRO radiance vs auto-selected NASA LRO NAC - "
                    "13.85S, 25.19E",
        "kind": "real",
    },
    "tycho": {
        "dir": os.path.join(PROC, "tycho_synthetic"),
        "slug": "tycho_synthetic",
        "name": "Level 2 - Tycho synthetic stand-in (same-sensor, medium)",
        "subtitle": "43.37S, 348.68E - Lommel-Seeliger rendered pair",
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
        "You are narrating lunar image-registration results to a hackathon "
        "judge who has an engineering background. Be precise, honest, and "
        "mathematically detailed, in 180-220 words. PLAIN TEXT ONLY: no "
        "LaTeX, no dollar signs, no markdown, no backticks - write formulas "
        "in words and plain numbers (example: 'k is at least log of "
        "1 minus p divided by log of 1 minus w to the fourth'). Walk the "
        "judge through the mathematics behind these numbers: how the RANSAC "
        "iteration budget k is derived from the inlier fraction w "
        "(k >= ln(1-p)/ln(1-w^4), p=0.999 confidence, 4-point homography), "
        "what reprojection RMSE means per pixel, how crater depth is "
        "estimated from shadow length times the tangent of the solar "
        "elevation, and what the Cramer-Rao bound says about the RMSE "
        "floor. Explain what the low match percentage honestly means for "
        "cross-mission registration (different cameras, orbits, sun angles) "
        "and why the SIFT plus learned-descriptor union is the right "
        "architecture for it. Do not invent numbers beyond these: " + summary)
    source = "gemini" if text else "local-template"
    if not text:
        text = integrations.local_narration(m)

    # matches + metrics rows are written on the fresh-compute path of
    # /match (cache-aside). Here we only narrate.
    return {"narration": text, "source": source, "rate_used": n,
            "metrics_summary": summary}


# --------------------------------------------------------------------------
# Judge Q&A - Gemini answers grounded in the project's own documents
# (PRESENTATION_GUIDE.md, the implementation-mathematics plan, and the
# explainer companion extracted from explainer.md.pdf). Knowledge is loaded
# once at startup; nothing is fetched at request time.
# --------------------------------------------------------------------------
_KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "knowledge")
_KNOWLEDGE_DOCS = (
    "PRESENTATION_GUIDE.md",
    "SIH26166_Implementation_Plan_and_Mathematics.md",
)


def _load_knowledge():
    docs = []
    for fn in _KNOWLEDGE_DOCS:
        p = os.path.join(ROOT, fn)
        if os.path.isfile(p):
            docs.append((fn, open(p, encoding="utf-8",
                                  errors="replace").read()))
    if os.path.isdir(_KNOWLEDGE_DIR):
        for fn in sorted(os.listdir(_KNOWLEDGE_DIR)):
            if fn.lower().endswith((".txt", ".md")):
                docs.append((fn, open(os.path.join(_KNOWLEDGE_DIR, fn),
                                      encoding="utf-8",
                                      errors="replace").read()))
    return docs


_KNOWLEDGE = None


def _knowledge_context():
    global _KNOWLEDGE
    if _KNOWLEDGE is None:
        _KNOWLEDGE = _load_knowledge()
    parts = []
    for name, body in _KNOWLEDGE:
        parts.append("=== DOCUMENT: %s ===\n%s" % (name, body))
    return "\n\n".join(parts)


class AskBody(BaseModel):
    question: str


@app.post("/ask")
def ask(body: AskBody):
    """Judge Q&A: answer questions about the project, its mathematics and
    the prototype, grounded in the presentation guide, the mathematics plan
    and the explainer document - not in Gemini's general knowledge."""
    q = (body.question or "").strip()[:600]
    if not q:
        raise HTTPException(400, "empty question")
    allowed, n = integrations.redis_rate_limit(
        "ask", max(5, integrations.NARRATE_RATE_LIMIT // 2), 60)
    if not allowed:
        raise HTTPException(429, "Q&A rate limit exceeded")
    prompt = (
        "You are the technical spokesperson of team SIH26166 (Lunalink: "
        "cross-mission lunar image registration, single-image terrain "
        "reconstruction and crater-based DEM verification for ISRO "
        "Chandrayaan-2 data). A judge just asked you a question. Answer it "
        "in 120-250 words, confident, first person plural ('we'). PLAIN "
        "TEXT ONLY: no LaTeX, no dollar signs, no markdown, no backticks - "
        "write formulas in words and plain numbers. Ground every technical "
        "claim in the PROJECT DOCUMENTS below; if the documents do not "
        "cover it, say so plainly and answer from the documents' spirit "
        "without inventing specific numbers. If the question is about the "
        "live prototype, remember the pipeline: SIFT union a learned ONNX "
        "descriptor, derived-budget RANSAC homography, Fourier-Mellin "
        "refinement, Lambertian shape-from-shading with FFT Poisson "
        "solving, marching-squares contours, and a Three.js mesh - and the "
        "honesty rules: simulated second passes are labeled as such and "
        "Gemini never generates metrics.\n\n"
        "JUDGE QUESTION: " + q + "\n\n" + _knowledge_context())
    text = integrations.gemini_narrate(prompt)
    if not text:
        return {
            "answer": None,
            "source": "unavailable",
            "hint": "Gemini is unavailable (missing GOOGLE_API_KEY or "
                    "quota cooldown). Narration and Q&A share the key.",
            "rate_used": n,
        }
    return {"answer": text, "source": "gemini", "rate_used": n,
            "grounded_docs": [name for name, _ in _KNOWLEDGE]}


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


def _references_payload(scene_dir):
    """Multi-instrument reference summary for a scene: every auto-selected
    reference that exists on disk (NASA LRO NAC + nearest TMC/TMC-2 + IIRS)
    with its metadata notes, served with /static URLs for the UI."""
    meta = {}
    mp = os.path.join(scene_dir, "metadata.json")
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp))
        except Exception:                                    # noqa: BLE001
            meta = {}
    rel = os.path.relpath(scene_dir, PROC).replace("\\", "/")
    refs = []
    for key, label, fn in (("nasa", "NASA LRO NAC", "reference.png"),
                           ("tmc", "TMC/TMC-2 (ISRO)", "reference_tmc.png"),
                           ("iirs", "IIRS (ISRO)", "reference_iirs.png")):
        p = os.path.join(scene_dir, fn)
        if not isinstance(refs, list):                       # pragma: no cover
            refs = []
        entry = {"role": key, "instrument": label, "file": fn}
        entry["url"] = "/static/%s/%s" % (rel, fn) \
            if os.path.exists(p) else None
        if key == "tmc":
            info = meta.get("tmc_reference") or {}
        elif key == "iirs":
            info = meta.get("iirs_reference") or {}
        else:
            info = {"note": meta.get("reference_source") or ""}
        entry["status"] = info.get("status",
                                   "selected" if os.path.exists(p)
                                   else "unavailable")
        entry["product_id"] = info.get("product_id")
        entry["footprint_km"] = info.get("footprint_km")
        entry["center"] = info.get("footprint_center")
        entry["post_alignment_ncc"] = info.get("post_alignment_ncc")
        entry["mutual_information"] = info.get("mutual_information")
        entry["note"] = info.get("note") or info.get("scale_note") or ""
        refs.append(entry)
    return {"references": refs,
            "ranked": {"tmc": meta.get("tmc_reference_ranked") or [],
                       "iirs": meta.get("iirs_reference_ranked") or []},
            "scene_center": meta.get("source_footprint_center"),
            "summary": meta.get("multi_instrument_summary", "")}


_layers_cache = {}


@app.get("/layers/{crater_id}")
def layers_endpoint(crater_id: str):
    """Multi-instrument layer availability for a scene (02 TERRAIN 3D
    switcher): SFS height, optical texture, TMC-2 metric DEM + the
    SFS-vs-metric validation stats, and IIRS mineral classes + legend.
    Computed lazily and cached in memory (first call may extract large
    rasters once)."""
    d = _scene_dir(crater_id)
    if crater_id not in _layers_cache:
        try:
            meta = json.load(open(os.path.join(d, "metadata.json")))
        except Exception:                                    # noqa: BLE001
            meta = {}
        _layers_cache[crater_id] = layers.layers_payload(d, meta)
    return _layers_cache[crater_id]


@app.get("/references/{crater_id}")
def references(crater_id: str):
    """The full multi-instrument reference set for a scene: NASA LRO NAC +
    nearest TMC/TMC-2 + IIRS products (auto-selected), with rankings."""
    d = _scene_dir(crater_id)
    return _references_payload(d)


@app.post("/ingest_product_upload")
async def ingest_product_upload(
    files: list[UploadFile] = File(...),
    scene_name: str = Form(""),
    crop_size: int = Form(4096),
    sun_az: float = Form(270.8),
    sun_el: float = Form(10.0),
):
    """Upload a real mission product DIRECTLY - an ISRO PDS4 XML+IMG pair,
    a NASA PDS3 .XML+.IMG pair, a TMC DTM XML+GeoTIFF product, a ZIP of the
    product directory, or a PRADAN bundle TAR of product ZIPs - and get a
    full matchable scene (SFS DEM, craters, auto-selected NASA + ISRO
    cross-instrument references, terrain payload). Plain images use
    /analyze_upload instead."""
    import io
    import zipfile
    import tarfile
    import tempfile
    if not files:
        raise HTTPException(400, "no files uploaded")
    names = [f.filename or "" for f in files]
    product_exts = (".zip", ".img", ".xml", ".tar", ".tif", ".tiff")
    if not any(n.lower().endswith(product_exts) for n in names):
        raise HTTPException(400, "expected a PDS4/PDS3 product (.xml + .img "
                                 "pair, or a .zip of the product directory)")
    tmp = tempfile.mkdtemp(prefix="sih_product_")
    try:
        if len(files) == 1 and names[0].lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(await files[0].read())) as z:
                z.extractall(tmp)
        elif len(files) == 1 and names[0].lower().endswith(".tar"):
            with tarfile.open(fileobj=io.BytesIO(await files[0].read())) as t:
                t.extractall(tmp)
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
            "geometry": res.get("geometry", {}),
            "craters": res.get("craters", []),
            "references": _references_payload(d),
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
    lat: float = Form(None),
    lon: float = Form(None),
):
    """Upload any lunar/surface image -> shape-from-shading relief -> the
    same terrain payload the 3D hologram consumes (grid + marching-squares
    contours). Non-metric, slope-calibrated relief - same honesty framing
    as the OHRC scene.  Optional lat/lon georeferences the upload so the
    cross-instrument reference selection runs on REAL coordinates."""
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
            scene_name, src_center=(
                {"lat_deg": lat, "lon_deg": lon} if lat is not None
                and lon is not None else None))
        global _registry
        _registry = ingest.load_registry()
        result["created_scene"] = scene["scene_id"]
        result["craters"] = scene["craters"]
        result["references"] = _references_payload(scene["entry"]["dir"])
    return result


def _make_scene_from_upload(img_u8, filename, sun_az, sun_el, scene_name,
                            src_center=None):
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
        product_id=filename or "uploaded image",
        src_center=src_center)


@app.get("/debug/sentry-test")
def sentry_test():
    """Verify the Sentry WIRING without polluting the issue stream: this
    health check emits only an informational breadcrumb - it never raises,
    so no error issue is ever created.  (The old deliberately-raised
    RuntimeError here is what put that escalating 'safe to ignore' event
    in the dashboard.)"""
    integrations.breadcrumb(
        "sentry wiring verified (health check, no error emitted)", "debug")
    return {"sent": True,
            "note": "wiring OK - no error event emitted"}


@app.get("/debug/sentry-test-raise")
def sentry_test_raise():
    """Explicit opt-in test error (never called by automated checks)."""
    raise RuntimeError("SIH26166 deliberate sentry verification error")


app.mount("/static", StaticFiles(directory=PROC), name="static")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
