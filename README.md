# SIH26166 — Lunar Scene Matching + 3D Terrain Hologram

Cross-illumination registration of **real Chandrayaan-2 OHRC** imagery against
an **auto-selected NASA LRO NAC reference**, with meter-scale shape-from-shading
relief reconstructed from a single image and a live 3-D polygon-mesh hologram.
Built for SIH 2026 problem SIH26166. Everything runs locally; every cloud
integration is optional and degrades to a no-op without keys. Total cost: $0.

> **Docs:** [`DEMO_GUIDE.md`](DEMO_GUIDE.md) — how to run it, the demo
> test-image set, the 60-second judge demo.
> [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md) — judge Q&A, the exhaustive
> system walkthrough, the ONNX training story, and the mathematical
> formulation of every stage.

## What it does

1. **Ingest** — parses any Chandrayaan-2 PDS4 product (XML label + IMG) or
   NASA PDS3 product (5064-byte attached header + int16 IMG), server-side or
   uploaded straight from the browser (pair or ZIP).
2. **Reconstruct relief** — linearized-Lambertian shape-from-shading turns one
   image into a meter-scale height field using the mission's own sun angles.
3. **Match** — SIFT + a custom-trained ONNX patch descriptor (union of both
   candidate branches) verified by RANSAC with a *derived* iteration budget,
   against a reference that is **selected automatically** from the LRO NAC
   library (or a simulated second pass when nothing overlaps).
4. **Detect craters** — Hough circles + shadow-projection depths, mapped to
   selenographic coordinates via the mission geometry file.
5. **Render** — a 72,962-triangle Three.js mesh of the Fourier-smoothed DEM
   with sub-pixel marching-squares contours.
6. **Narrate** — Gemini turns the computed metrics into judge-friendly words,
   with a local-template fallback; it never invents numbers.

## Architecture

```
browser (zero-build: 1 HTML file, vanilla ES modules, Three.js via import map)
   │  fetch /health · /craters · /match/{id} · /terrain/{id} · /narrate/{id}
   │  POST /analyze_upload (image) · /ingest_product_upload (PDS4/PDS3 pair|zip)
   ▼
FastAPI (uvicorn, single process serves API + frontend)
   ├─ pipeline.py      SIFT + ONNX union match, RANSAC/DLT, Fourier-Mellin,
   │                   truncated-Fourier smoothing, marching squares
   ├─ preprocess.py    PDS4 label + .spm sun file + memmap IMG → SFS DEM,
   │                   LRO NAC 2-stage registration, crater detection
   ├─ ingest.py        dynamic scene ingestion + registry + auto-reference
   ├─ lroc.py          NASA LRO NAC CDR ingest (coarse NCC → template + SIFT)
   └─ integrations.py  Supabase · Upstash Redis · Sentry · Gemini (all optional)
Data on disk: data/raw (PRADAN products) · data/reference/lro_nac (8 NASA
strips) · data/processed/<scene>/ (source.png, reference.png, dem.npy,
craters.json, metadata.json) · data/processed/registry.json.
Committed subsets (fresh clones are demo-ready): the three demo scenes
under data/processed/, registry.json, data/demo_upload/, and
backend/models/descriptor.onnx — the 1 GB raw products and 3.8 GB LRO
library stay local.
```

## Run

**Demo day: the Vercel link.** The frontend is deployed on Vercel; the
backend runs on Render (free tier) from the included `render.yaml`
Blueprint, and the two are wired by one variable,
`frontend/config.js:window.API_BASE`. Full one-time setup:
[`DEMO_INSTRUCTIONS.md` §14](DEMO_INSTRUCTIONS.md). Pre-warm the backend
(open `<backend>/health` in a tab) — the free tier sleeps after ~15 min idle.

**Local (offline fallback / development):**

```bash
cd backend
python preprocess.py              # one-time: 1 GB .img → DEM + match pair
python -m uvicorn main:app --port 8000
# open http://localhost:8000  (backend serves the frontend too)
```

The three demo scenes (`data/processed/{ohrc_real,tycho_synthetic,demo_tmc}`),
their `registry.json`, the demo upload set (`data/demo_upload/`), and the
trained `backend/models/descriptor.onnx` are all committed — a fresh clone
is demo-ready without the 1 GB raw products.

Retrain the descriptor: `python backend/train_descriptor.py` (CPU, minutes).
Optional keys in `backend/.env` (see `.env.example`): Supabase, Upstash,
Sentry, Gemini. The app is fully functional with none of them.
## Dataset (Stage 0/1, on disk)
`data/UNZIPPED_DATA/ch2_ohr_ncp_20210401T2357376656_d_img_d18/`
- **Image**: 90,148 × 12,000 px 8-bit panchromatic, GSD 0.265 m/px, alt 104.22 km,
  acquired 2021-04-01T23:57:37Z, descending orbit, footprint 13.06°S–13.89°S /
  25.13°E–25.25°E
- **Label** (PDS4 XML): focal length 2080 mm, 5.2 µm pixels, TDI64, exposure 181.74 ms
- **Geometry CSV**: per-pixel lon/lat ground coordinates
- **Sun file (.spm)**: sun elevation 9.92° → incidence 80.08°, azimuth 270.8°
  (near-terminator imaging — exactly the shadow-robustness challenge)
- **Reference library**: `data/reference/lro_nac/` — 8 real NASA LRO NAC CDR
  strips (~0.5 m/px), scanned automatically for overlap with any scene

## API endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | liveness + which matcher is loaded (SIFT-only vs SIFT + learned ONNX) |
| `GET /craters` | scene registry with full parsed metadata (feeds the dropdown + metadata panel) |
| `GET /match/{id}` | full match payload: keypoints, inliers, homography, RMSE, derived RANSAC budget, method breakdown (SIFT vs ONNX), Fourier-Mellin diagnostic, craters |
| `GET /terrain/{id}` | Stage-7 payload: 192×192 Fourier-smoothed height grid + 8 marching-squares contour levels |
| `GET /narrate/{id}` | Redis rate-limited Gemini narration of the computed metrics (local-template fallback) |
| `POST /analyze_upload` | plain image → SFS relief → terrain payload; `make_scene=true` promotes it to a full matchable scene with auto-selected reference |
| `POST /ingest_product_upload` | ISRO PDS4 XML+IMG pair, NASA PDS3 pair, or a ZIP of the product directory → full matchable scene |
| `POST /ingest_product` | server-side path ingestion of any `data/raw` product, any crop window |
| `GET /static/...` | processed scene imagery (source/reference PNGs) |
| `GET /debug/sentry-test` | deliberate Sentry event to verify ingestion |

## How the backend works (request lifecycle)

`GET /match/{id}` is the hot path and shows every pattern:

1. **Cache-aside**: in-memory dict → Upstash Redis `GET match:{id}` (TTL'd;
   >900 KB payloads are trimmed before caching) → else fresh compute.
2. **Fresh compute**: load `source.png` + `reference.png` + `dem.npy` →
   CLAHE photometric normalization → SIFT (8000 kp, Lowe ratio 0.8) →
   learned ONNX branch (embed → cosine > 0.5 → mutual NN) → union →
   RANSAC + DLT homography → RMSE → Fourier-Mellin diagnostic.
3. **Persistence**: Redis `SET` (TTL), then Supabase rows into
   `scenes` / `jobs` / `matches` / `metrics` (RLS: public read only on
   scenes + metrics; writes go through the backend-only service key).
4. **Concurrency**: sync endpoints run in Starlette's threadpool, so
   CPU-bound inference never blocks the event loop; uploads are `async`.
5. **Narration**: `GET /narrate/{id}` → Redis sorted-set sliding window
   (20/min) → metrics summary built ONLY from computed values → Gemini
   (`gemini-flash-lite-latest`, free tier) → ≤120-word explanation.
   Fallbacks: 429 → one warning + 5-minute cooldown; any failure →
   local-template narration with identical numbers.
6. **Error tracking**: Sentry on both sides (browser project + backend
   project, breadcrumbs per pipeline stage, `/debug/sentry-test`).
7. **Frontend boot is resilient**: health failures retry with backoff,
   the status badge is click-to-retry, and no skeleton can stick.

## Scenes (dropdown) & automatic reference selection

1. **CH-2 OHRC real scene** — real ISRO radiance; relief from shape-from-shading
   (non-metric, slope-calibrated, stated in metadata). Reference: the real NASA
   LRO NAC strip **M1249388815LC**, auto-selected from the 8-strip library and
   registered to the OHRC grid.
2. **Tycho (synthetic stand-in)** — Kaggle-notebook DEM, Lommel–Seeliger pair at
   two sun geometries with regolith albedo texture.
3. **Any dynamically ingested scene** — uploaded images, or direct PDS4/PDS3
   product uploads — each gets a reference chosen automatically:

`auto-select pipeline` (preprocess.py for the baked scene, ingest.auto_select_reference
for everything else): template-match the 1024² source inside an 8×-downsampled
preview of **every** LRO NAC strip (TM_CCOEFF_NORMED) → register each candidate
at full resolution (translation pre-alignment, then SIFT + RANSAC homography
refinement) → keep the best post-alignment NCC → write the decision into
`metadata.json:reference_source` (the UI caption displays it). If no strip
overlaps, a **simulated second pass** (gamma 1.35, rotation 1.8°, scale 1.035,
radiance gradient + noise) is generated so the scene is always matchable —
and labeled as simulated, never passed off as real.

## The ONNX descriptor — trained in-repo (`backend/train_descriptor.py`)

Self-supervised, no labels, runs on CPU in minutes:

1. **Data**: three procedural crater DEMs (parabolic bowl + Gaussian rim +
   central peak + 5-octave fractional-Brownian roughness) with layered
   3-scale Gaussian albedo fields.
2. **Triplets**: the same terrain rendered at *random* sun geometries —
   anchor (az θ, el 15–40°), positive (az θ+20…60°, el 30–60°), negative
   (different DEM/albedo). The network learns terrain identity, not
   illumination — the exact failure mode of SIFT descriptors across
   sun passes.
3. **Model**: 4-block CNN (16→32→64→128 ch) → AdaptiveAvgPool →
   Linear(128) → L2-normalize.
4. **Loss**: triplet hinge `mean(clamp(‖a−p‖₂ − ‖a−n‖₂ + m, 0))`, m = 0.2,
   Adam lr 1e-3.
5. **Validation**: 100 fresh triplets, ranking accuracy `‖a−p‖ < ‖a−n‖`
   printed before export.
6. **Export**: `torch.onnx.export`, opset 17, dynamic batch axis —
   float32→float32, op-for-op, no quantization → **no accuracy loss**;
   onnxruntime then verifies shape (1,128) and unit L2 norm in the same
   script.
7. **Inference** (`pipeline.py::_embed`): top-1500 keypoints per image →
   edge-padded 128×128 patches → batched embeddings → cosine > 0.5 →
   mutual-NN filter → unioned with SIFT candidates before RANSAC. Any
   inference failure degrades ONCE (warning event) to the documented
   SIFT-only fallback — the endpoint cannot 500 because of the model.

## Honesty framing (non-negotiable)

- The relief for the real scene is a photometric approximation, **not** a
  metric DEM — the metadata panel and every scene's `metadata.json` say so
  verbatim.
- References are labeled by origin: `REAL NASA LROC NAC — AUTO-SELECTED` vs
  `AUTO-GENERATED SECOND PASS`. Simulated numbers never masquerade as real.
- Cross-mission match percentages are low *by nature* (different missions,
  sun geometries, radiometry) — that difficulty IS the research problem;
  the narrator explains it rather than hiding it.
- Gemini narrates metrics the pipeline computed; it is never allowed to
  generate numbers.
- `negative_control_low_feature.png` in the demo set proves the pipeline
  does not hallucinate relief when the image carries no signal.

## Mathematical formulation (per stage)

Full derivations in [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md);
all formulas below are implemented verbatim in the referenced functions.

- **Shape-from-shading** (`preprocess.py::shape_from_shading`): Lambertian
  `I ≈ A(−p·s_x − q·s_y + s_z)`; a single image constrains only the
  sun-parallel slope → minimum-norm slope `t = −ΔI/(a‖s_h‖²)`, shadow-masked,
  clipped to tan 35° → FFT Poisson integration `ĥ = F[∇·p]/(−k²)` (DC guard,
  Hanning window) → radiometric calibration to an 8° RMS slope (reported,
  never claimed as metric).
- **Craters** (`detect_craters`): Hough circles + shadow-projection depth
  `d ≈ L·tan(θ_sun)`, selenographic mapping via the geometry CSV.
- **Matching**: SIFT (DoG + sub-pixel Taylor refinement, Lowe 0.8) union
  learned branch (cosine > 0.5, mutual NN).
- **Verification** (`match_pair`): 4-point DLT homography in RANSAC with a
  **derived** budget `k ≥ log(1−p)/log(1−w⁴)` (s = 4, p = 0.99, w from the
  scene's own inlier fraction) → reprojection RMSE over inliers; the RMSE
  floor is the image's gradient-information ceiling (Cramér–Rao intuition).
- **Fourier–Mellin** (`fourier_mellin_coarse`): log-magnitude spectra →
  log-polar resample → phase correlation → rotation θ = Δ∠·360/N_θ, scale
  s = exp(Δlnρ·ln(maxR)/N_ρ), then translation by a second phase
  correlation (Reddy–Chatterji).
- **Mutual information** (`mutual_information`, IIRS stretch goal):
  `I(A;B) = Σ p(a,b) log[p(a,b)/p(a)p(b)] ≥ 0` (Jensen).
- **Truncated Fourier surface** (`fourier_smooth`, Problem 7): keep only
  low-|m|,|n| FFT coefficients — exact low-pass; periodic scan artifacts
  die in a single (m,n) bin.
- **Marching squares** (`marching_squares`, Problem 8): sub-pixel contour
  crossing `t = (z_k − z₁)/(z₂ − z₁)`, `P = P₁ + t(P₂ − P₁)`.

## The 3-D hologram

`THREE.PlaneGeometry(1000, 1000, 191, 191)` = **36,864 vertices /
72,962 triangles**, each vertex displaced by the Stage-7 height grid, plus
a wireframe overlay and 8 contour levels as 3-D line segments. Vertical
exaggeration is computed from the DEM's own z-span so the display never
exaggerates silently.

## Demo assets — `data/demo_upload/` (committed to the repo)

Real OHRC crop (PNG + fast JPG), synthetic feature-rich crater field, a
256 px quick version, a **negative control** (flat terrain — proves no
hallucinated relief), and a PDS4 product pair + ZIP for the direct product
upload demo. The folder is committed (gitignore exception) so every clone
is demo-ready. See [`DEMO_GUIDE.md`](DEMO_GUIDE.md) and
[`DEMO_INSTRUCTIONS.md`](DEMO_INSTRUCTIONS.md) — the full feature-by-feature
demo playbook.

## Repo layout

```
frontend/index.html        the entire frontend (zero build)
backend/main.py            FastAPI endpoints (match/terrain/narrate/uploads)
backend/pipeline.py        match + RANSAC + Fourier-Mellin + Stage-7 math
backend/preprocess.py      PDS4 ingest, SFS DEM, LRO NAC registration, craters
backend/ingest.py          dynamic scene ingestion + auto reference selection
backend/lroc.py            NASA LRO NAC CDR parser + registration
backend/integrations.py    Supabase / Upstash / Sentry / Gemini (optional)
backend/train_descriptor.py  ONNX descriptor training + export + verification
backend/models/descriptor.onnx   the trained artifact (gitignored)
supabase/schema.sql        tables + RLS policies
render.yaml                Render Blueprint — one-click backend deploy
audit.py / _sanity_check.py  end-to-end verification scripts (accept a
                             deployed backend URL as argv[1])
data/demo_upload/          demo test images (committed)
```

## Requirements & troubleshooting

```bash
pip install fastapi uvicorn numpy opencv-python scipy pillow onnxruntime torch httpx
```
Frontend needs internet once (Three.js CDN via import map).

- **Status badge red on the Vercel page** → backend asleep (hit `/health`,
  wait ~50 s), or `window.API_BASE` in `frontend/config.js` doesn't point
  at the Render backend — fix and `vercel --prod`. Full deployment setup:
  [`DEMO_INSTRUCTIONS.md` §14](DEMO_INSTRUCTIONS.md).
- **Status badge red in the local demo** → start uvicorn
  (`python -m uvicorn main:app --port 8000` in `backend/`), then click the
  badge — boot retries automatically. If you host the HTML elsewhere
  (Live Server), set `window.API_BASE` in `frontend/config.js`.
- **Deployed backend reports `learned_model_loaded: false`** → the ONNX
  artifact didn't reach the host; it is committed at
  `backend/models/descriptor.onnx` — check the Render build log.
- **Narration says local-template** → Gemini quota exhausted; it resets
  daily, and the fallback carries identical numbers.
- **First match on a scene is slow** → it's the fresh compute + LRO NAC
  auto-selection scan; results cache in memory and Redis afterwards.

