# PRESENTATION GUIDE — SIH26166 (Lunalink)

Judge-facing Q&A, the exhaustive system walkthrough, how the ONNX
descriptor was trained, and the mathematical formulation of every stage.
Everything here is grounded in the actual code — file/function references
included so you can defend any claim live.

**Architecture in one line:** zero-build vanilla JS + Three.js frontend
(static on **Vercel**), FastAPI backend (container host on **Render**, from
the repo's `render.yaml` Blueprint), OpenCV SIFT + custom ONNX learned
descriptor + RANSAC matching, FFT shape-from-shading DEMs, auto-selected
NASA LROC NAC references, Supabase/Upstash/Sentry/Gemini as *optional*
no-op-without-keys integrations. Total cost: $0.

---

## Part 1 — Cross questions: technologies, APIs & packages

### 1. Why React over Vue/Angular for this use case?
Neither — a deliberate decision, not an omission. The frontend is a single
`frontend/index.html` with vanilla ES modules + Three.js (import map, CDN).
For a four-view dashboard whose only shared state is one scene ID, a
component framework adds a build chain and npm supply-chain risk with zero
benefit. The backend serves the frontend itself — one process for the whole
demo. React's core value is managing complex shared state; ours is one
variable that triggers a re-match.

### 2. What build/dev advantage does Vite give over Webpack here?
We went one step further: **no bundler at all**. Native ES modules +
import maps mean no build step, no dev server, no HMR, no `node_modules`
(check `.gitignore`). Nothing to configure or break on demo morning.
Advantage over *any* bundler: the deployment artifact is one HTML file.

### 3. What are you actually rendering in 3D — analytical or visual?
A real polygon mesh of *computed* terrain: `THREE.PlaneGeometry(1000,
1000, 191, 191)` = **36,864 vertices / 72,962 triangles**, each vertex
Y-displaced by the Stage-7 DEM grid (192×192 heights), plus a wireframe
overlay and 8 marching-squares contour levels as 3-D line segments.
Analytical value: relief-range readout (0–91 m on the OHRC scene), crater
rim/bowl morphology, contour spacing as a slope map, and a *computed*
vertical exaggeration so the display never lies about relief.

### 4. How does FastAPI's async handling manage concurrent, GPU-bound inference?
Honest answer: our inference is **CPU-bound**, and FastAPI handles it via
its threadpool — sync `def` endpoints run in Starlette's threadpool
(~40 workers) so concurrent `/match` requests never block the event loop;
`async def` is used where we genuinely await I/O (multipart uploads). The
real concurrency strategy is **caching**: in-memory dict → Upstash Redis
(TTL'd) → fresh compute, so identical requests never re-run SIFT.

### 5. Did you convert PyTorch models to ONNX yourself? Accuracy loss?
Yes — `torch.onnx.export` (opset 17, dynamic batch axis) in
`backend/train_descriptor.py`, and no measurable loss: the export is a
float32→float32, op-for-op graph translation with **no quantization or
pruning**. The script verifies the artifact end-to-end in onnxruntime
(output shape (1,128), L2 norm ≈ 1.0 asserted) and prints triplet-ranking
accuracy before export. Model: 4-block CNN (16→32→64→128 ch), 128-D
L2-normalized embeddings, triplet loss (margin 0.2), trained on
synthetically shaded DEM pairs at randomized sun geometries.

### 6. ONNX Runtime GPU (CUDA) or CPU?
CPU, explicitly: `providers=["CPUExecutionProvider"]`. The workload is
128×128 patch embeddings — sub-millisecond on CPU. CUDA would add a hard
dependency for zero visible gain; switching is a documented one-line change.

### 7. Which PyTorch version, any extensions?
torch **2.11.0 (CPU build)**. Pure `torch` + `torch.nn` — no Lightning, no
torchvision. Hand-rolled training loop (Adam, lr 1e-3, batched triplets);
data pipeline is NumPy + PIL. The same script runs locally on CPU and is
Kaggle-T4-ready for the full-scale model.

### 8. Why Supabase over raw PostgreSQL — and RLS?
Managed Postgres keeps real SQL (jsonb columns for footprints and 3×3
homographies) without ops, plus an auto REST API. **RLS is genuinely
enabled on all four tables** (`supabase/schema.sql`): public read policies
on `scenes`/`metrics`, deliberately **no** public policy on
`jobs`/`matches` — those write only through the service-role key, which
lives only in `backend/.env` (gitignored), never in the frontend.

### 9. Does Supabase Storage handle large GeoTIFFs efficiently?
No — and that's the honest answer. Multi-MB rasters never enter the DB
layer: processed scenes live on the server filesystem under
`data/processed/<scene>/` (source.png, reference.png, dem.npy,
craters.json, metadata.json), gitignored and regeneratable. Only small
JSON rows go to Supabase (keypoint samples capped at 500). Production
answer: PDS archives + object storage. Upstash's ~1 MB value cap is
exactly why big payloads stay out of Redis too (the cache trims
`matches` before writing).

### 10. Is Upstash's HTTP Redis fast enough for real-time job queuing?
We don't use it as a work queue — we use it for what HTTP-Redis is good
at: cache-aside payloads, job status keys, and rate limiting, all of which
tolerate ~5–20 ms REST latency because they sit in front of a pipeline
that takes seconds anyway. For hard real-time queuing we'd run traditional
Redis; the code isolates it behind four small helper functions in
`integrations.py`, so swapping is trivial.

### 11. Which Redis data structures for job management?
Two: **strings** (`SET/GET` + TTL) for the match cache and job status, and
a **sorted set** sliding window (`ZADD` → `ZREMRANGEBYSCORE` → `ZCARD` →
`PEXPIRE`) for the per-minute narration limiter — a true sliding window,
not a fixed-window counter. No lists/streams/pub-sub: minimal structure
per need.

### 12. Sentry for frontend, backend, or both?
Both, as separate projects: browser SDK (DSN in `frontend/config.js` —
safe to expose client-side) and `sentry_sdk` on the backend with
breadcrumbs after each pipeline stage, plus a deliberate
`/debug/sentry-test` endpoint to verify ingestion.

### 13. Why Gemini over GPT/Claude?
Cost (free tier, no card — the whole stack runs at $0), latency
(flash-lite class for a ≤120-word single-turn summary; context window is
irrelevant at this task size), and framing: Gemini is prompt-locked to
metrics the pipeline computed — it never generates numbers.

### 14. Fallback if Gemini is down or rate-limited?
Three layers, all live-tested: (1) local-template narration carrying the
same metrics — the UI labels the source (`gemini` vs `local-template`);
(2) a 429-aware 5-minute cooldown (`integrations.py::_gemini_cooldown_until`)
so a dead quota emits ONE warning event, not an error per click;
(3) a Redis sliding-window limiter (20/min) upstream.

### 15. What blocks a fully on-premise/offline ISRO deployment?
Nothing is a hard blocker — every integration is a graceful no-op without
its key (first guarantee in `integrations.py`). Swap paths: Supabase →
vanilla Postgres (our schema.sql is plain SQL), Upstash → local Redis
(4 REST helpers → redis-py), Sentry → self-hosted GlitchTip or off,
Gemini → the built-in template narration or a local LLM. The *data* is
already local (PRADAN PDS4 products + NASA LRO NAC strips on disk) —
nothing core phones home.

### 16. Walk me through your deployment (live, right now).
Two pieces, one wire:
- **Frontend → Vercel.** The whole UI is one static HTML file
  (`vercel.json` sets `outputDirectory: frontend`) — zero build, so the
  deployment artifact IS the source. Vercel gives the global CDN.
- **Backend → Render.** Python + OpenCV + scene data can't live in
  serverless functions, so the FastAPI app runs as a web service deployed
  from the repo's `render.yaml` Blueprint (build: `pip install -r
  requirements.txt`, start: `uvicorn main:app --host 0.0.0.0 --port $PORT`,
  health check `/health`). The three demo scenes and the trained
  `descriptor.onnx` are committed, so a fresh deploy boots complete.
- **The wire.** One variable: `window.API_BASE` in `frontend/config.js`
  points the Vercel page at the Render URL; CORS on the backend is
  `allow_origins=["*"]`. No cookies, no sessions — stateless JSON + static
  imagery under `/static`.
- **Cost: $0** (Vercel Hobby + Render free tier + Gemini free tier). The
  free tier sleeps after ~15 min idle — a cost-control trade-off, not
  latency: first request warms it in ~50 s, and the pipeline itself runs in
  seconds with cache hits after.
- **Why this proves portability:** the same backend boots unchanged on a
  laptop (`uvicorn main:app --port 8000`), on Render, or on ISRO
  infrastructure — nothing core phones home (Q15). Demo-day reality check:
  the live demo runs the frontend from Vercel with the backend on our own
  laptop behind a free HTTPS tunnel (`start_demo_tunnel.ps1`) — no cold
  starts, and the full 3.8 GB NASA reference library stays live so fresh
  uploads get real NASA auto-selection; Render remains the always-on
  fallback URL (Q on cold starts below).


---

## Part 2 — Exhaustive system walkthrough (what happens, step by step)

**① Site opens** → uvicorn serves `frontend/index.html` (`GET /`) →
`config.js` (API_BASE, Sentry DSN) → Sentry browser SDK inits → import map
loads Three.js 0.160 → `GET /health` turns the badge green and reports
"SIFT + learned ONNX descriptor" → `GET /craters` fills the scene dropdown
(each entry = the scene's `metadata.json`: product ID, sun angles,
footprint lat/lons, DEM range) → auto-runs the match on the default scene.

**② MATCH — `GET /match/{scene_id}`** → cache-aside: in-memory dict →
Upstash Redis `GET match:{id}` → else fresh compute: load `source.png` +
`reference.png` + `dem.npy` → CLAHE normalization → SIFT (8000 kp, Lowe
ratio 0.8) → **learned ONNX branch** (top-1500 kp per image → 128×128
patches → 128-D embeddings → cosine > 0.5 + mutual nearest neighbor) →
union → **RANSAC + DLT homography** (3 px, p = 0.99, derived k) → RMSE →
Fourier-Mellin diagnostic → persist (Redis SET, Supabase rows) → frontend
draws keypoint/inlier overlays + crater table → `refCap` caption shows
WHICH reference was auto-picked.

**③ Reference auto-selection (no manual step anywhere)** — for the baked
scene, `preprocess.py` template-locates the OHRC crop inside all 8 LRO NAC
strips and keeps the best post-alignment NCC (chose `M1249388815LC`).
For any upload/product, `ingest.auto_select_reference()` runs the same
scan live; if no strip overlaps, it auto-generates a **simulated second
pass** (gamma 1.35, rotation 1.8°, scale 1.035, radiance gradient + noise)
so every scene is matchable. The choice is recorded in
`metadata.json:reference_source`, which the UI caption reads.

**④ UPLOAD (plain image)** → `POST /analyze_upload` (file + sun az/el,
auto-fit from mission `.spm` metadata) → PIL decode → 1024² →
shape-from-shading → terrain payload → client-side hillshade →
SEND TO TERRAIN 3D / CREATE MATCHABLE SCENE (SFS + crater detection +
auto-reference + registration → new scene appears in MATCH).

**⑤ UPLOAD (dataset formats)** → `.xml + .img` pair or `.zip` →
`POST /ingest_product_upload` → `_read_product()` parses ISRO PDS4 or
NASA PDS3 (5064-byte attached header, int16) labels → memmap →
best-variance crop → 1024² grid → the same SFS/craters/auto-reference
chain → scene auto-selected.

**⑥ TERRAIN — `GET /terrain/{id}`** → `dem.npy` → truncated-Fourier
low-pass → 192×192 grid → marching squares at 8 levels → JSON →
`buildTerrain()`: 72,962-triangle mesh + wireframe + 3-D contour polylines
+ computed vertical exaggeration + OrbitControls.

**⑦ NARRATE — `GET /narrate/{id}`** → Redis sliding window (20/min) →
metrics summary compiled ONLY from computed values → Gemini
(`gemini-flash-lite-latest`) → ≤120-word judge-facing narration → local
template fallback; 429 → cooldown.

**⑧ Errors** → Sentry (frontend JS project + backend Python project,
breadcrumbs per stage). Raw artifacts for every stage are inspectable on
disk: `data/processed/<scene>/` + `registry.json`.

## Part 3 — How we trained the ONNX descriptor (end to end)

Script: `backend/train_descriptor.py` → artifact: `backend/models/descriptor.onnx`.
Run: `python backend/train_descriptor.py` (CPU, ~minutes; env knobs
`N_PAIRS`, `EPOCHS`, `N_TRIALS`). Same script is Kaggle-T4-ready for the
full-scale model.

**1. Training data — self-supervised, no labels needed.**
Three synthetic crater DEMs are generated procedurally:
`DEM(x,y) = bowl + rim + peak + roughness`, where the bowl is a parabolic
depression `−D·clip(1−(r/r_rim)², 0, 1)`, the rim a Gaussian ring
`H_r·exp(−(r−r_rim)²/2σ²)`, an optional central peak, plus 5-octave
fractional-Brownian roughness. An albedo field (3-scale Gaussian random
field, clipped ±35 %) is layered on top.

**2. Shaded pair generation — the cross-illumination problem, baked in.**
Each training triplet is rendered with a Lambertian shading model at
RANDOM sun geometries: anchor at azimuth θ, elevation 15–40°; the positive
sample is the SAME terrain at azimuth θ+20…60° and elevation 30–60°; the
negative is a different DEM/albedo combination. The network therefore
learns to identify *terrain*, not illumination — exactly the failure mode
plain SIFT descriptors suffer between different sun passes.

**3. Model.** `PatchEncoder`: Conv(1→16) → ReLU → MaxPool → Conv(16→32) →
Conv(32→64) → Conv(64→128) → AdaptiveAvgPool → Linear(128) → L2-normalize.
Output: a unit-length 128-D embedding where terrain-matched patches land
close together regardless of lighting.

**4. Loss — triplet hinge.**
`L = mean( clamp( ‖a−p‖₂ − ‖a−n‖₂ + m, 0 ) )`, margin m = 0.2.
Optimizer: Adam, lr 1e-3, batches of 8 triplets.

**5. Validation before export.** 100 fresh random triplets; metric =
ranking accuracy `‖a−p‖ < ‖a−n‖` (target > 0.5, ideally > 0.9). Printed,
not assumed.

**6. ONNX export.** `torch.onnx.export(..., opset_version=17,
dynamic_axes={"patch": {0: "batch"}, "embedding": {0: "batch"}})` —
dynamic batch so the server can embed in batches of 256. **No
quantization/pruning → float32 → float32, op-for-op: no accuracy loss.**

**7. Post-export verification (in the same script).** onnxruntime runs the
artifact: asserts output shape (1, 128) and |‖embedding‖₂ − 1| < 0.01.
The deployment pipeline additionally reports learned-branch candidates in
every `/match` payload (`method_breakdown.learned_candidates`), so the
model's live contribution is always measurable.

**8. Inference integration** (`pipeline.py::_embed` + `match_pair`):
top-1500 response keypoints per image → edge-padded 128×128 patches →
batched embedding → cosine similarity matrix → threshold 0.5 → mutual
nearest-neighbor filter → unioned with the SIFT branch before RANSAC.
If inference ever fails, the branch degrades ONCE with a warning event to
the documented SIFT-only fallback — the demo cannot 500 because of the
model.

## Part 4 — Mathematical formulation, stage by stage

Every formula below is implemented verbatim in the referenced function.

### Stage 2 — Shape-from-shading (`preprocess.py::shape_from_shading`)
Lambertian reflectance with sun unit vector **s** = (s_x, s_y, s_z) in the
(east, south, up) image frame (azimuth is compass-clockwise from north,
converted at `sun_vector_image_frame`):

    I(x,y) ≈ A · ( −p·s_x − q·s_y + s_z ),   (p, q) = (∂h/∂x, ∂h/∂y)

A single image constrains only the slope component parallel to the sun's
horizontal projection s_h, so we take the **minimum-norm solution**:
with A·s_z estimated robustly as `a = median(I)/s_z`, define ΔI = I − a·s_z
and

    t = −ΔI / (a·‖s_h‖²),      (p, q) = −(s_x, s_y) · t

with |t| clipped to tan 35° and shadow pixels (I below the 1st percentile)
masked to zero — shadows carry no slope information. The slope field is
then integrated with an **FFT Poisson solve**:

    ∇²h = ∂p/∂x + ∂q/∂y   ⇒   ĥ(k) = F[div](k) / (−k²),
    k² = (2πf_x)² + (2πf_y)²,  k²[0,0] := 1 (DC guard), Hanning-windowed

A single image cannot fix absolute height (albedo is entangled with
slope), so the relief is radiometrically calibrated to a stated plausible
RMS slope of 8°: `h *= tan(8°)/RMS(‖∇h‖)` — reported in metadata, never
presented as metric. **Honesty is part of the math.**

### Crater detection + depth (`preprocess.py::detect_craters`)
Hough circle detection on the CLAHE-enhanced grid (radii 5–90 px), then
shadow-projection depth: a rim shadow of length L at sun elevation θ
implies `depth ≈ L·tan(θ)` (flat-floor approximation). Positions map to
selenographic lat/lon via bilinear lookup in the mission geometry CSV.

### Stage 4 — Candidate matching (`pipeline.py::match_pair`)
SIFT keypoints (DoG + sub-pixel Taylor refinement) with the Lowe ratio
test `d_m < 0.8·d_n`. Learned branch: cosine similarity matrix
`S = E₁E₂ᵀ` over L2-normalized 128-D embeddings; keep `S_ij > 0.5` where
i and j are **mutual** nearest neighbors. The two branches are UNIONED —
SIFT contributes gradient-precision, the learned branch illumination
robustness.

### Stage 5 — RANSAC with a derived budget (`pipeline.py::match_pair`)
4-point DLT homography inside RANSAC (threshold 3 px, confidence
p = 0.99). The iteration budget is **derived from the matcher's own
inlier fraction w**, never hard-coded (Problem 3):

    k ≥ log(1 − p) / log(1 − wˢ),   s = 4 (minimal homography sample)

Quality = inlier count + reprojection **RMSE** over inliers:
`RMSE = √(mean ‖H·xᵢ − xᵢ′‖²)`. When RMSE stops improving, that is the
image's own gradient-information ceiling (Cramér–Rao intuition:
Var ≥ σ²/Σ(dI/dx)²) — a physical floor, not a pipeline defect.

### Fourier–Mellin coarse registration (`pipeline.py::fourier_mellin_coarse`)
Reddy–Chatterji: translation → rotation+scale decoupling in the frequency
domain. Log-magnitude spectra `A = log|FFT(a)|` are resampled to log-polar
(360 angles × 256 log-radii); **phase correlation**
`(Δ∠, Δln ρ) = argmax ℱ⁻¹( P_A·P_B* / |P_A·P_B*| )` gives rotation
`θ = Δ∠·360/N_θ` and scale `s = exp(Δlnρ·ln(maxR)/N_ρ)`; after
de-rotating/de-scaling, a second phase correlation yields the translation.
Reported as a diagnostic — fine registration remains SIFT + RANSAC.

### Problem 6 — Mutual information (`pipeline.py::mutual_information`)
    I(A;B) = Σ p(a,b) · log[ p(a,b) / (p(a)p(b)) ] ≥ 0 (Jensen: log is
    concave, equality iff independence)
Implemented as a joint-histogram sum — the cross-modal similarity
statistic reserved for the IIRS stretch goal.

### Problem 7 — Truncated 2-D Fourier surface (`pipeline.py::fourier_smooth`)
    z(x,y) = Σ c_mn · exp( i2π(mx/Lx + ny/Ly) ),  c_mn = FFT2 coefficients
Keeping only low-|m|,|n| terms is an exact low-pass (macroscopic terrain
shape survives; pixel noise dies), and a periodic scan-line artifact
concentrates in one (m,n) bin — zeroing that bin removes it without any
spatial-domain estimation. This coefficient grid IS the mesh input.

### Problem 8 — Marching squares, sub-pixel (`pipeline.py::marching_squares`)
On each cell edge crossed by contour level z_k, linear interpolation

    t = (z_k − z₁)/(z₂ − z₁),   P = P₁ + t·(P₂ − P₁)

places the contour at sub-pixel precision; corners are connected per the
cell's sign pattern (same as `skimage.find_contours`).

### Reference auto-selection statistics
Overlap scoring = normalized cross-correlation (template matching,
TM_CCOEFF_NORMED) at a coarse 8× preview of each LRO NAC strip, followed
by full-resolution registration (translation pre-alignment + SIFT/RANSAC
homography refinement); the winner is the best post-alignment NCC
(the baked scene selected `M1249388815LC`). Cross-mission consistency of
the final pair is quantified as raw Pearson + gradient-magnitude
correlation between the independent LROC NAC observation and the OHRC
shape-from-shading relief — recorded in `metadata.json`, reported
honestly (low values are *expected* across missions and sun geometries;
that difficulty is the research problem itself).



