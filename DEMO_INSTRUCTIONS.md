# DEMO INSTRUCTIONS — SIH26166 (Lunalink)

**The complete, feature-by-feature playbook for demoing the project to judges.**
Every instruction here is grounded in the real code — file/function names are
included so you can defend any claim when a judge asks "how?".

> Companion docs: [`DEMO_GUIDE.md`](DEMO_GUIDE.md) (quick run + test images) ·
> [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md) (judge Q&A, ONNX training
> story, full math) · [`README.md`](README.md) (architecture).

**What the project is, in one sentence:** we take a single image of the lunar
surface (from ISRO's Chandrayaan-2 OHRC, a NASA product, or any image you
upload), reconstruct meter-scale 3-D relief from it using physics, register it
against an automatically-selected NASA LRO reference using a matcher that
combines classical SIFT with a neural descriptor we trained ourselves, detect
craters with shadow-based depth estimates, render the terrain as an
interactive 3-D hologram, and have Gemini narrate the numbers for judges —
while never letting the AI generate a single number.

---

## 0 · Pre-demo checklist (T−30 minutes)

**You will demo from the Vercel link.** The Vercel deployment is the
frontend only — it talks to the backend over HTTPS (see §14 to deploy the
backend on Render with the included `render.yaml`, and to wire
`frontend/config.js` → `window.API_BASE`). Do this checklist against the
VERCEL URL, not localhost:

1. **Wake the backend FIRST** (Render free tier spins down after ~15 min
   idle; the first request after a spin-down takes ~50 s). Open
   `https://<your-backend>.onrender.com/health` in a browser tab until it
   returns `{"status":"ok","learned_model_loaded":true,...}`. Keep that tab
   open — it keeps the service warm.
2. **Open the Vercel link** — the status badge must turn
   `backend online · SIFT + learned ONNX descriptor` (green = CORS +
   `API_BASE` wired correctly). If it stays red: click it (retries with
   backoff), and check §15.

Then verify against the Vercel page:

| Check | Expected |
|---|---|
| Status badge (top right) | `backend online · SIFT + learned ONNX descriptor` |
| Scene dropdown (01 MATCH) | lists `CH-2 OHRC real scene`, `Tycho (synthetic stand-in)`, `demo_tmc` |
| First match auto-runs | match ring fills, keypoints overlay on both canvases |
| 02 TERRAIN | mesh auto-rotates; drag orbits, scroll zooms |
| 03 UPLOAD → drop `data/demo_upload/ch2_ohrc_real_crop_web.jpg` | relief shading appears in ~5–15 s |
| 04 NARRATION → click the button | `[Gemini] …` (or `[local template]` if quota is out — say it's a designed fallback, identical numbers) |
| Backup | screen-record the full demo beforehand; also keep a local `python -m uvicorn main:app --port 8000` fallback running on the demo laptop |

**Network needs at the venue:** the Vercel page loads Three.js from a CDN
and the backend must be reachable — test on the venue Wi-Fi/hotspot. If the
venue network is unusable, fall back to the local demo:
`cd backend && python -m uvicorn main:app --port 8000` → open
`http://127.0.0.1:8000` (everything works offline except Gemini narration).

---

## 1 · The demo image folder — `data/demo_upload/`

This folder is **committed to the repository** (gitignore exception), so a
fresh `git clone` is demo-ready. During the demo, open File Explorer at this
folder and drag files straight into the 03 UPLOAD dropzone:

| File | When to use it |
|---|---|
| `ch2_ohrc_real_crop_web.jpg` (346 KB) | **The main upload demo.** Real ISRO Chandrayaan-2 OHRC crop (0.265 m/px). It overlaps a real NASA LRO strip, so the reference is auto-selected from the NASA library — the strongest story. |
| `ch2_ohrc_real_crop_1024.png` | Same scene, lossless PNG — use if a judge asks about compression artifacts. |
| `synthetic_craters_feature_rich.png` | Synthetic crater field with obvious craters — makes shape-from-shading and the 3-D mesh visually undeniable. Best fallback for a crisp number. |
| `synthetic_craters_256_fast.jpg` (21 KB) | Fastest possible upload if the demo laptop is slow. |
| `negative_control_low_feature.png` | **The honesty demo.** Flat, low-contrast terrain — the pipeline reports near-zero relief, proving it does NOT hallucinate when there is no signal. Judges love this. |
| `demo_tmc_product_crop_1024.png` | Image cut from a Chandrayaan-2 TMC-style PDS4 product — "product → image" story. |
| `product_pds4_pair/` (`ch2_tmc_demo_d_img_d18.xml` + `.img`) | **Direct product upload.** Select BOTH files together in the file dialog (or drag both). The backend parses the PDS4 label and builds a full matchable scene automatically. |
| `product_pds4_demo.zip` | Same product as a ZIP — one-drag version of the above. |
| `reference_AUTO_selected_LROC_NAC.png` | **Not for upload** — it is the auto-selected real NASA LRO NAC reference, for showing on a slide next to the source image. |

---

## 2 · Demo scripts

### The 60-second version

1. **01 MATCH** (already auto-run): point at the match ring — "this is a REAL
   cross-mission match: ISRO Chandrayaan-2 OHRC against NASA LRO, matched
   with SIFT plus a neural descriptor we trained, verified by RANSAC. The
   number is honest — different missions, different suns, different cameras."
2. **02 TERRAIN**: "this mesh is computed from a SINGLE image by
   shape-from-shading, using the mission's own sun angles. Drag to orbit."
3. **03 UPLOAD**: drag `ch2_ohrc_real_crop_web.jpg` → relief appears →
   "one image in, 3-D terrain out" → SEND TO TERRAIN 3D.
4. **04 NARRATION**: click narrate → "Gemini turns the metrics into words —
   it only narrates numbers our pipeline computed; it is never allowed to
   generate them."

### The 3-minute deep version

Add: the negative control upload (honesty), the PDS4 ZIP product upload
(full mission-product ingestion), the AUTO→MANUAL sun slider demo
(physics interactivity), and the method-breakdown panel (SIFT vs learned
candidates + derived RANSAC budget).

---

## 3 · Window 01 — MATCH (every element explained)

| UI element | What it does | What to say |
|---|---|---|
| **Scene dropdown** | Every scene = a folder under `data/processed/` with `metadata.json` (product ID, sun angles, footprint, DEM range). Selecting a scene auto-runs its match. | "Scenes are self-describing — the dropdown is generated from the mission metadata, not hard-coded." |
| **REAL ISRO DATA / SYNTHETIC STAND-IN tag** | The `.tag` shows provenance (`metadata.json:provenance`). Synthetic Tycho numbers are never mixed with real-scene numbers. | "We label provenance on-screen. The synthetic scene exists only as a stand-in; the flagship scene is real ISRO data." |
| **▶ RUN MATCH** | `GET /match/{id}` → cache-aside (in-memory → Redis → fresh compute) → the whole Stage 3–6 pipeline. First run computes; later runs are instant (cache). | "First click computes for real; the cache layer is shown right there in the panel." |
| **Match ring (inverted black card)** | Inlier ratio, animated SVG ring. | "Inliers = keypoints that survive geometric verification. This is the single headline number." |
| **inliers / RMSE px / kp source / kp reference / cache layer** | Inlier count; reprojection RMSE over inliers (sub-pixel on good scenes); keypoint counts on both images; which cache tier answered. | "RMSE is the geometric error of the homography over the surviving matches — measured in pixels." |
| **🔊 NARRATE FOR JUDGES** | `GET /narrate/{id}` → Gemini, narrating ONLY the computed metrics. Rate-limited to 20/min by a Redis sliding window. | See section 7. |
| **Method breakdown** (muted panel) | SIFT candidates vs **learned (ONNX)** candidates vs the **derived** RANSAC budget `k ≥ log(1−p)/log(1−w⁴)`. | "The RANSAC iteration count is derived live from the matcher's own inlier fraction — never hard-coded." |
| **Scene metadata** (muted panel) | Product ID, acquisition time, band, footprint lat/lon, GSD, sun elevation/azimuth, incidence angle, DEM range, provenance note. | "Full traceability. Every number on screen comes from the mission's own metadata files." |
| **Correspondences canvas** | Left = SOURCE (CH-2 OHRC), right = REFERENCE (caption says WHICH reference was auto-picked, e.g. `REAL LROC NAC — AUTO-SELECTED`). White dots = matches; bright/large = RANSAC inliers, faint = rejected. | "Bright dots survived geometric verification; faint dots were rejected. The matcher shows its own work." |
| **Crater identification** | Hough circles overlaid on the source + a table of `r=…m @ (lat°, lon°) · shadow …m · depth ≈ …m`. Depth from shadow length × tan(sun elevation), positions from the mission geometry CSV. | "We turn 2-D shadows into 3-D depths using the sun angle, and map every crater to selenographic coordinates." |

## 4 · Window 02 — TERRAIN 3D (the hologram)

| UI element | What it does | What to say |
|---|---|---|
| **3-D viewport** | `THREE.PlaneGeometry(1000, 1000, 191, 191)` = **36,864 vertices / 72,962 triangles**, each vertex displaced by the Stage-7 height grid, over a dark stage with a starfield. | "This is not a texture trick — it is a real polygon mesh of computed terrain." |
| **Drag / scroll** | OrbitControls: drag to orbit, scroll to zoom, auto-rotating when idle. | "Fully interactive — judges can fly over the terrain themselves." |
| **White lines** | 8 marching-squares contour levels rendered as 3-D line segments, sub-pixel interpolated. | "Contours are a slope map: tight lines = steep rim, wide spacing = flat floor." |
| **Wireframe overlay** | Semi-transparent triangle wireframe over the mesh. | "You can see the actual tessellation we compute." |
| **Readout line below** | `192×192 grid · cell ≈ 5.3 m · extent ≈ 1.02 km · relief 0–85 m · vertical exaggeration ×N · source: <scene>` | "Vertical exaggeration is computed from the DEM's own span and printed — the display never exaggerates silently." |

How the mesh gets there: `GET /terrain/{id}` → `dem.npy` → truncated-Fourier
low-pass → 192×192 grid → marching squares at 8 levels → JSON → mesh.

---

## 5 · Window 03 — UPLOAD (from a single image to a 3-D hologram)

This is the "wow" window and the easiest to demo live.

### The flow, click by click

1. Drag `ch2_ohrc_real_crop_web.jpg` from `data/demo_upload/` onto the
   dropzone (or click it and pick the file).
2. ~5–10 s later the right canvas shows **RECONSTRUCTED RELIEF SHADING** and
   the key-value panel reports relief range, grid size, sun angles, contour
   levels.
3. **⬡ SEND TO TERRAIN 3D** → switches to 02 TERRAIN with your uploaded
   image's mesh. **⚡ CREATE MATCHABLE SCENE** → promotes the upload to a
   full scene: it appears in the 01 MATCH dropdown and runs the complete
   pipeline (SFS + crater detection + auto reference selection + registration).

### What "RECONSTRUCTED RELIEF SHADING" actually is

The backend ran **linearized-Lambertian shape-from-shading**
(`backend/preprocess.py::shape_from_shading`, exposed via
`POST /analyze_upload`): it estimates slope at every pixel from brightness
using the physics of how a Lambertian surface reflects sunlight, integrates
the slope field into a height map with an FFT Poisson solve, then re-renders
the height map as a hillshade. The right canvas is a re-illumination of
*reconstructed 3-D geometry*, not a filter on the photo — drag the sun
sliders and watch the shading physically re-render (the hillshade is
re-computed client-side from the height grid in real time,
`renderHillshade()` in `frontend/index.html`).

### How the sun azimuth auto-update works (say this verbatim — it's a differentiator)

- The sun-parameter file shipped with the mission product (`.spm` for
  Chandrayaan-2; `parse_sun_angles()` in `backend/preprocess.py`) records the
  REAL sun azimuth and elevation at the moment of capture (for our OHRC
  product: az 270.8°, el 9.9° — near-terminator imaging).
- When a scene is selected, the frontend reads those values from the scene's
  `metadata.json` and the UPLOAD window's sliders **move by themselves** —
  the `AUTO · fitted from scene mission metadata` line shows the fitted
  values (`window.__setSun` in `frontend/index.html`).
- The physics therefore uses the *actual* sun, not an assumption — this is
  why the reconstructed relief matches reality. Dragging either slider
  switches the label to `MANUAL override` and re-renders the relief live, so
  you can demonstrate that the shading responds to sun geometry like real
  terrain would (shadows swing around as azimuth rotates).

### Uploading a full mission product (PDS4/PDS3)

Drop the `.zip` (or select the `.xml` + `.img` pair together) →
`POST /ingest_product_upload` → the backend parses the ISRO PDS4 label
(or a NASA PDS3 5064-byte attached-header product), memory-maps the int16
image, picks the best-variance 1024² crop, runs the full SFS → craters →
auto-reference → registration chain, and auto-selects the new scene in 01
MATCH. One drag: raw mission product to matchable, registered scene.

---

## 6 · How matching works (the 90-second explanation)

1. **Normalize** — CLAHE on both images (the OHRC and LRO NAC have very
   different radiometry).
2. **SIFT branch** — up to 8000 keypoints per image, Lowe ratio 0.8. SIFT is
   gradient-based: precise localization, illumination-tolerant to a degree.
3. **Learned ONNX branch** — top-1500 keypoints per image → 128×128 pixel
   patches → our CNN (`descriptor.onnx`) embeds each patch into a 128-D
   vector → cosine similarity > 0.5 + mutual nearest neighbor. This branch
   was TRAINED to recognize terrain across illumination changes (see §8).
4. **Union** — the two candidate sets are merged; SIFT contributes gradient
   precision, the learned branch contributes illumination robustness.
5. **RANSAC + 4-point DLT homography** — 3 px threshold, p = 0.99, iteration
   budget **derived** from the observed inlier fraction w: `k ≥ log(1−p)/
   log(1−w⁴)`. Survivors = inliers; RMSE over inliers is the quality number.
6. **Fourier-Mellin diagnostic** — frequency-domain rotation/scale estimate
   (Reddy–Chatterji) reported as a cross-check.
7. **Reference auto-selection** — nobody picks the reference. For any scene,
   the system scans the 8-strip NASA LRO NAC library (coarse NCC template
   match → full SIFT/RANSAC refinement) and keeps the best
   post-alignment correlation; if nothing overlaps, it generates a simulated
   second pass (gamma 1.35, 1.8° rotation, 1.035 scale, noise) so every
   scene is matchable. The choice is recorded in `metadata.json` and shown
   in the reference caption.

## 7 · What role ONNX plays (and how we trained it)

**ONNX = Open Neural Network Exchange** — a runtime-independent format for
neural networks. Our model is a CNN exported from PyTorch with
`torch.onnx.export` (opset 17) and executed with **onnxruntime on CPU**
(`providers=["CPUExecutionProvider"]`) — 128×128 patch embeddings are
sub-millisecond, so CUDA would add a hard dependency for zero gain.

**What it does in the pipeline:** it is the second matcher. SIFT describes
keypoints with hand-crafted gradient histograms; our `descriptor.onnx`
embeds the raw 128×128 pixel patch around each keypoint into a 128-D vector
such that *patches of the same terrain look identical even under different
illumination*. The two candidate sets are unioned before RANSAC — you can
see each branch's live contribution in the Method breakdown panel
(`SIFT candidates` vs `learned (ONNX)`).

**How it was trained (self-supervised, no labels):**
`backend/train_descriptor.py` (same script runs on CPU locally or a Kaggle
T4 for the full-scale model):
1. Generate synthetic crater DEMs procedurally
   (`DEM(x,y) = bowl + rim + peak + roughness`).
2. Render Lambertian-shaded image triplets at RANDOM sun geometries: anchor
   at azimuth θ / elevation 15–40°, positive = SAME terrain at θ+20…60° /
   elevation 30–60°, negative = different terrain.
3. Train a 4-block CNN (16→32→64→128 ch) with **triplet loss** (margin 0.2)
   so anchor-to-positive distance < anchor-to-negative distance.
4. Export to ONNX; the script re-loads the artifact in onnxruntime and
   asserts output shape (1,128) and ‖embedding‖₂ ≈ 1.0.

**Why this is the point of the project:** the network was *forced* to learn
"same terrain, different light" — exactly the cross-mission failure mode of
classical matching. And it is engineered for graceful degradation: if
inference ever fails, the pipeline logs one warning and continues SIFT-only
(`/health` shows `learned_model_loaded: true/false` — check it in the demo).

## 8 · The mathematics of terrain reconstruction (per stage, with code refs)

Every formula is implemented verbatim in the referenced function —
`PRESENTATION_GUIDE.md` Part 4 has the full derivations.

**Shape-from-shading** (`preprocess.py::shape_from_shading`).
Lambertian reflectance with sun unit vector **s** = (s_x, s_y, s_z) in the
(east, south, up) image frame (azimuth compass-clockwise from north,
converted in `sun_vector_image_frame`):

    I(x,y) ≈ A · ( −p·s_x − q·s_y + s_z ),   (p, q) = (∂h/∂x, ∂h/∂y)

One image constrains only the slope component parallel to the sun's
horizontal projection, so we take the **minimum-norm solution**:
with `a = median(I)/s_z` and `ΔI = I − a·s_z`,

    t = −ΔI / (a·‖s_h‖²),      (p, q) = −(s_x, s_y) · t

|t| clipped to tan 35°, shadow pixels masked (no slope information there).
Then the slope field is integrated with an **FFT Poisson solve**:

    ∇²h = ∂p/∂x + ∂q/∂y   ⇒   ĥ(k) = F[div](k) / (−k²),  k²[0,0] := 1

**Radiometric (non-metric) calibration:** a single image cannot separate
albedo from slope, so the relief is scaled to a stated plausible RMS slope
of 8°: `h *= tan(8°)/RMS(‖∇h‖)` — reported in metadata, never claimed as a
metric DEM. **Honesty is part of the math.**

**Crater depth** (`preprocess.py::detect_craters`): Hough circles on the
CLAHE grid, then shadow projection `depth ≈ L·tan(θ_sun)`; positions mapped
to selenographic lat/lon via bilinear lookup in the mission geometry CSV.

**Truncated Fourier surface** (`pipeline.py::fourier_smooth`): keep only
low-|m|,|n| FFT coefficients — an exact low-pass; macroscopic terrain
survives, pixel noise dies, and a periodic scan artifact concentrates in a
single (m,n) bin and can be zeroed without any spatial estimation.

**Marching squares** (`pipeline.py::marching_squares`): on each cell edge
crossed by level z_k, linear interpolation
`t = (z_k − z₁)/(z₂ − z₁)`, `P = P₁ + t·(P₂ − P₁)` places contours at
sub-pixel precision.

**RANSAC budget** (`pipeline.py::match_pair`): `k ≥ log(1−p)/log(1−w⁴)`
with s = 4 (minimal homography sample), p = 0.99, w = the scene's own
inlier fraction — derived, never hard-coded. When RMSE stops improving, that
is the image's gradient-information ceiling (Cramér–Rao intuition:
Var ≥ σ²/Σ(dI/dx)²) — a physical floor, not a pipeline defect.

**Fourier–Mellin** (`pipeline.py::fourier_mellin_coarse`): log-magnitude
spectra → log-polar resample (360 angles × 256 log-radii) → phase
correlation gives rotation and scale; a second phase correlation after
de-rotation gives translation (Reddy–Chatterji). Reported as a diagnostic.

**Mutual information** (`pipeline.py::mutual_information`):
`I(A;B) = Σ p(a,b)·log[p(a,b)/(p(a)p(b))] ≥ 0` — the cross-modal similarity
statistic reserved for future multi-sensor (IRS-class) registration.

## 9 · How Gemini narration works

`GET /narrate/{id}` → **Redis sorted-set sliding-window rate limit**
(ZADD → ZREMRANGEBYSCORE → ZCARD, 20/min) → a metrics summary built **only**
from values the pipeline computed → Gemini (`gemini-flash-lite-latest`,
free tier) → ≤120-word judge-facing explanation. The UI prefixes the source:
`[Gemini]` or `[local template]`.

**The guardrail (say this to judges):** Gemini narrates; it is never allowed
to generate numbers. The prompt receives only computed metrics, and every
fallback layer carries the *identical* numbers:
- quota/429 → ONE warning event + a 5-minute cooldown
  (`integrations.py::_gemini_cooldown_until`), then the local template
  narrates with the same metrics;
- no API key at all → local template from the start — the demo never breaks.

---

## 10 · How the whole program works (end to end, one breath)

**① Boot** — one uvicorn process serves BOTH the API and the zero-build
frontend (`GET /` → `frontend/index.html`). No bundler, no npm: vanilla ES
modules + Three.js via an import map. The deployment artifact for the
frontend is literally one HTML file.

**② Health & scenes** — `GET /health` (reports whether the ONNX model is
loaded) turns the badge green; `GET /craters` fills the dropdown from scene
metadata; the default scene's match auto-runs.

**③ Ingest** — three equivalent doors into the same pipeline:
`POST /analyze_upload` (any image), `POST /ingest_product_upload`
(ISRO PDS4 XML+IMG / NASA PDS3 / ZIP), or server-side `POST /ingest_product`
from `data/raw`. Every door ends in the same scene folder structure:
`source.png, reference.png, dem.npy, craters.json, metadata.json`.

**④ Reconstruct** — shape-from-shading turns one image into a DEM using the
mission's own sun angles (§8).

**⑤ Match** — SIFT ∪ learned-ONNX candidates → derived-budget RANSAC
homography → RMSE → Fourier-Mellin diagnostic, against an auto-selected
NASA LRO NAC reference (§6).

**⑥ Detect** — Hough craters + shadow-projection depths + selenographic
mapping.

**⑦ Render** — truncated-Fourier smoothing → 192×192 grid → marching-squares
contours → 72,962-triangle Three.js mesh with 3-D contour lines (§4).

**⑧ Narrate & persist** — Gemini narrates the computed metrics (rate-limited,
with fallback); Supabase rows (`scenes/matches/metrics`, RLS-enabled) and
Upstash Redis cache/status keys are written **if keys exist** — every
integration is a graceful no-op without keys, total cost $0.

**⑨ Observe** — Sentry breadcrumbs after every pipeline stage (frontend +
backend as separate projects) so a silent bug surfaces as an event, not a
wrong number.

## 11 · Why this is novel, useful, and impactful

**Novel**
1. **A trained cross-illumination descriptor for lunar matching.** We didn't
   fine-tune someone else's checkpoint — we generated synthetic DEMs,
   rendered Lambertian triplets at randomized sun geometries, trained a
   triplet-loss CNN, exported it to ONNX, and unioned it with SIFT. The
   network learns "same terrain, different light" — the exact failure mode
   of cross-mission matching.
2. **Auto reference selection.** No human picks the reference image; the
   system scans the NASA library and chooses (or synthesizes a second pass
   when nothing overlaps). The whole registration loop is hands-free.
3. **Derived, not tuned.** The RANSAC budget is computed from the matcher's
   own statistics; vertical exaggeration is computed from the DEM's own
   span; the Cramér–Rao-style RMSE floor is explained, not hidden.
4. **Honesty as an architecture principle.** Provenance tags, non-metric
   relief labeled as such, a negative-control image proving the pipeline
   doesn't hallucinate, and an LLM that is forbidden from generating numbers.
5. **Physics + learning in one loop.** Shape-from-shading (physics) gives
   3-D from one image; the learned descriptor (ML) gives matching
   robustness; each covers the other's weakness.

**Useful** — landing-site characterization, crater depth/diameter statistics
from shadows, registration of any new lunar image against a reference
library, and an ingest path that already speaks ISRO PDS4 **and** NASA PDS3.

**Impactful** — it is **data-source-agnostic and sovereign-data friendly**:
everything runs locally on a laptop, every cloud piece is optional, the
stack costs $0, and it works offline except narration/3-D-viewer niceties.
Swap in ISRO's real TMC-2/OHRC archive and it runs unchanged — that claim is
demonstrated live, not promised.

## 12 · Future upgrades — how TMC-2 and IRS imagery take this further

**TMC-2 (Chandrayaan-2 Terrain Mapping Camera-2, 5 m/px, stereo triplets):**
1. **Metric DEMs replace calibrated relief.** TMC-2's fore/aft/nadir stereo
   enables photogrammetric DEMs with true heights. Today our OHRC relief is
   radiometrically calibrated to an 8° RMS slope (honest but non-metric);
   with TMC-2 DEMs we anchor the same SFS output to *metric* heights —
   a one-line scale change, already flagged in `metadata.json`.
2. **Multi-resolution fusion** — OHRC (0.265 m/px) supplies meter-scale
   detail; TMC-2 supplies km-scale context. Our truncated-Fourier
   representation is natively band-limited, so fusion = coefficient
   blending across scales.
3. **True stereo validation** — compare our single-image SFS against TMC-2
   stereo DEMs of the same terrain: a quantitative accuracy number for the
   photometric method itself.
4. **TMC-2 hillshades as a reference library** — the auto-selection scanner
   (§6 step 7) works on any image library; adding TMC-2 hillshades widens
   reference coverage beyond the current 8 LRO NAC strips.

**IRS-class sensors (CARTOSAT series, Resourcesat, etc.) and other missions:**
1. **The pipeline already speaks NASA PDS3** (5064-byte attached header +
   int16 IMG) — the parser layer, not the algorithm, is per-mission. Adding
   another sensor = one more label parser.
2. **Cross-modal registration** — the mutual-information matcher
   (`pipeline.py::mutual_information`, §8) is implemented and waiting as the
   similarity statistic for multi-sensor pairs (optical ↔ spectral ↔ DEM),
   which is exactly the IRS/ISRO multi-instrument case.
3. **Terrestrial extension** — nothing in the math is Moon-specific except
   constants; desert/canyon terrain from CARTOSAT DEMs runs the same
   hologram pipeline.
4. **Scale-out** — the same ONNX descriptor can be retrained on
   mission-specific shaded renderings in hours (the generator is in the
   training script), making the matcher sensor-aware without changing
   the deployment.

## 13 · Data sources & provenance (be ready to name them)

| Source | What we use | Where |
|---|---|---|
| **ISRO PRADAN** — Chandrayaan-2 OHRC PDS4 product | 90,148×12,000 px panchromatic, 0.265 m/px, acquired 2021-04-01, footprint 13.06°S–13.89°S / 25.13°E–25.25°E; PDS4 XML label + `.spm` sun file + geometry CSV | `data/UNZIPPED_DATA/ch2_ohr_ncp_…/` |
| **NASA PDS** — LRO NAC CDR strips | 8 real reference strips; `M1249388815LC` is the auto-selected reference for the flagship scene | `data/reference/lro_nac/` |
| **Chandrayaan-2 TMC-style demo product** | PDS4 XML+IMG pair for the direct product-upload demo | `data/demo_upload/product_pds4_pair/` |
| **Google AI Studio** | Gemini API key (free tier) — narration only | `GOOGLE_API_KEY` in `backend/.env` |
| **Supabase / Upstash / Sentry** | optional persistence / cache+rate-limit / error tracking (all no-op without keys) | `backend/.env` |
| **Kaggle** (training-time only) | GPU for the full-scale descriptor training | `backend/train_descriptor.py` |

## 14 · Deployment — the Vercel + Render setup (do this ONCE, before demo day)

**The one fact to remember:** Vercel hosts the **frontend only**
(`vercel.json`: `outputDirectory: "frontend"`). Python + OpenCV + scene
data do not fit serverless, so the **backend lives on Render** (free tier
works). The two talk over HTTPS; CORS on the backend is already
`allow_origins=["*"]`.

Everything the backend needs at boot is **committed to this repo**: the
three demo scenes (`data/processed/ohrc_real`, `tycho_synthetic`,
`demo_tmc` + `registry.json`) and the trained model
(`backend/models/descriptor.onnx`, 0.44 MB). A fresh clone boots complete.

### One-time setup (~20 minutes)

**A. Deploy the backend on Render (no manual config — the Blueprint does it):**
1. Push this repo to GitHub (it already is — `render.yaml` is at the root).
2. **render.com** → sign in with GitHub → **New → Blueprint** → pick the
   `Lunalink` repo → Render reads `render.yaml` → **Apply**.
3. Wait for the build (`pip install -r requirements.txt`, then
   `uvicorn main:app --host 0.0.0.0 --port $PORT`). Note the URL, e.g.
   `https://sih26166-backend.onrender.com`.
4. Verify: open `https://<backend-url>/health` → must say
   `"learned_model_loaded": true`. If it says false, the ONNX file didn't
   make it — check the Render build log for git errors.
5. (Optional, dashboard → Environment) add `GOOGLE_API_KEY` for live Gemini
   narration, and optionally Supabase/Upstash/Sentry keys. Without them
   everything still works: narration falls back to the local template.

**B. Point the Vercel frontend at the backend:**
1. Edit `frontend/config.js` → set
   `window.API_BASE = "https://<backend-url>";` (the exact URL from A.3,
   no trailing slash).
2. From the repo root: `vercel --prod` (or push to GitHub — if the Vercel
   project is connected to the repo, it redeploys automatically).
3. Open the Vercel URL → the status badge must turn green within ~60 s.

**C. Verify the deployed pair end-to-end** (from any machine):
- 01 MATCH auto-runs on `CH-2 OHRC real scene` ✓
- 02 TERRAIN renders ✓
- 03 UPLOAD with `data/demo_upload/ch2_ohrc_real_crop_web.jpg` ✓
- 04 NARRATION ✓
- `python _sanity_check.py <backend-url>` from a clone also exercises the
  whole API against the deployed backend.

### Two behavior differences on the deployed backend (be ready to explain)

1. **Cold start / spin-down (Render free tier).** After ~15 min idle the
   service sleeps; the next request takes ~50 s to boot. **Pre-warm before
   the demo** (§0 step 1) — open `/health` in a tab and keep it open. Say
   to judges: "that's the free tier's cost control, not our latency — the
   pipeline itself runs in seconds, and it's cache-warm after the first
   match."
2. **New uploads get a simulated second-pass reference, not a NASA strip.**
   The 3.8 GB LRO NAC library stays on the team machine (it cannot ship to
   GitHub). The three committed scenes carry their REAL auto-selected NASA
   reference (`M1249388815LC`), so the flagship match is still
   source-vs-NASA. For a fresh upload the auto-selector finds no local
   library and generates the documented **simulated second pass** (gamma,
   radiance gradient, noise, slight rotation/scale) — recorded verbatim in
   the scene metadata and shown in the reference caption. This is the
   graceful-degradation design working exactly as documented, on two
   continents' worth of infrastructure differences.


## 15 · Troubleshooting during the demo

| Symptom | Meaning / fix |
|---|---|
| Red status badge `backend offline — CLICK TO RETRY` on the Vercel page | (1) Backend asleep — hit `/health` once and wait ~50 s, click the badge. (2) `window.API_BASE` in `frontend/config.js` empty or wrong → set it to the Render URL, `vercel --prod`. (3) Backend not deployed → §14 step A. |
| First request after idle takes ~50 s | Render free-tier spin-down — pre-warm per §0; not pipeline latency. |
| Health says `learned_model_loaded: false` on Render | `descriptor.onnx` missing from the deploy — check the Render build log; the file is committed at `backend/models/descriptor.onnx`. |
| Uploads on the deployed backend say `SIMULATED SECOND PASS` in the reference caption | Expected — the 3.8 GB LRO NAC library stays local; committed scenes keep their REAL NASA reference. Explain per §14. |
| Narration says `[local template]` | Gemini quota exhausted (resets daily) or `GOOGLE_API_KEY` not set in the Render environment — the fallback carries identical numbers; say "designed fallback" and move on. |
| First match on a scene is slow (5–15 s) | Fresh compute + registration scan; results cache afterwards (cache tier is shown in the panel). |
| Terrain panel says TERRAIN UNAVAILABLE | Backend restarted/spun down after the page loaded — click the status badge to re-boot. |
| Canvases say "image unavailable" | Backend `/static` not reachable — same root cause as the red badge. |
| Venue Wi-Fi blocks/hinders the demo | Fallback: local backend `cd backend && python -m uvicorn main:app --port 8000` → `http://127.0.0.1:8000` (offline-capable except Gemini). |
| Red status badge in the LOCAL demo | Start uvicorn, then click the badge — boot retries with backoff automatically; the page never sticks on "loading…". |



