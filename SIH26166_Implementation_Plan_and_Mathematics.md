# SIH26166 — Exhaustive Implementation Plan + Required Mathematics

Target: Tycho crater, 43.37°S, 348.68°E — one crater, one reference pair, one demo DB.

---

## PART 1 — Implementation Plan

### Stage 0 — Data acquisition (do this first, has the longest lead time)
1. Register on `pradan.issdc.gov.in` on day one (manual approval delay).
2. Browse `chmapbrowse.issdc.gov.in`, footprint-search around 43.37°S, 348.68°E, note the
   OHRC/TMC-2 orbit/scene ID + acquisition date (needed for sun-angle metadata).
3. Confirm actual reference coverage on `quickmap.lroc.im-ldi.com` for that same bounding
   box *before* pulling anything full-resolution — OHRC's ~12×3 km footprint means a small
   coordinate error can mean zero real overlap.
4. Pull the PDS4-labelled source product (`.xml` + `.img`) from PRADAN once approved.
5. Pull the matching LROC NAC product id from `lroc.im-ldi.com`.
6. Pull an SLDEM2015 tile (Tycho is within ±60° latitude, so this beats plain LOLA) from
   the PDS Geosciences Node, crop to footprint immediately.
7. Pull the Tycho entry + surrounding rim points from the LPI/USGS crater catalog as an
   external geometric anchor.
8. Verify the Makharia et al. 2025 ISRO SAC citation (exact title/DOI) before it's on a slide.

Folder layout: `data/raw/{ohrc,tmc2,iirs}/<scene_id>/`, `data/reference/lro_nac/<product_id>/`,
`data/dem/<region>_sldem2015.tif`, `data/crater_catalog/<region>_craters.csv`,
`data/processed/{synthetic_pairs,metrics}/`. Never mutate `raw/`/`reference/` in place.

### Stage 1 — Ingest (`src/ingest/`)
- Read PDS4 label + `.img` via `rasterio`/GDAL (OHRC/TMC-2), band-order-aware cube read for
  IIRS, `rasterio`/GDAL for the LROC NAC product.
- Parse orbit/attitude ephemeris and sensor interior-calibration values out of the PDS4
  label metadata — these feed Stage 3's collinearity equation directly; never estimated
  from pixels.

### Stage 2 — Geometry (`src/geometry/`)
- `pixel_to_ground.py` / `ground_to_pixel.py`: collinearity ray–DEM intersection (Problem 9).
- `triangulate.py`: TMC-2 fore/nadir/aft stereo-triplet ray intersection to build your own
  DEM patch when an external one is thin for the exact scene (Problem 4).
- Reproject the reference image into the source image's frame using the same collinearity
  machinery, so Stage 5 compares like-for-like.

### Stage 3 — Photometric correction (`src/photometric/`)
- Normalize illumination/intensity offsets between source and reference before matching
  (this is exactly the failure mode SIFT is documented to break on).
- Gaussian pyramid construction here must be **blur, then downsample** (Problem 10) — get
  the operator order wrong and coarse-to-fine correspondences break silently.

### Stage 4 — Matching (`src/matching/`) — SIFT + learned descriptor, in parallel
- `sift_refine.py`: DoG keypoint detection, then sub-pixel Taylor-expansion refinement
  (Problem 1) so reported match locations aren't locked to the integer pixel grid.
- `pyramid.py`: coarse-to-fine search across the ~17–20:1 OHRC-to-TMC-2 scale ratio.
- Learned branch: `descriptor.onnx` (trained per Part 2 of the previous message) embeds
  patches around each SIFT keypoint; Euclidean/cosine distance thresholded into candidate
  correspondences — this is the shadow-robust branch.
- `mutual_information.py`: optional cross-modal similarity statistic for the IIRS stretch
  goal (Problem 6) — build the joint intensity histogram, normalize, sum directly.
- Union both branches' candidate correspondences; log each branch's contribution
  separately (`method_breakdown` in the `/match` response) rather than one merged number.

### Stage 5 — Verify (`src/verify/`) — RANSAC + homography
- `dlt_homography.py`: direct linear transform via SVD null-space (Problem 2).
- `ransac.py`: `estimate_iterations(w, s, p)` derived from the inlier-fraction probability
  bound (Problem 3) — log the estimated inlier fraction per scene so the iteration budget
  is a stated, derived number, not a magic constant.
- Output: inlier mask, homography matrix, inlier count/ratio → this is your reported
  **percentage match**.

### Stage 6 — Evaluate (`src/evaluation/`)
- RMSE, inlier count, inlier ratio, per logged run — never asserted, never synthetic
  numbers mixed with real-scene numbers.
- Cite the Cramér–Rao bound (Problem 5) when RMSE approaches the image's own local-gradient
  information ceiling — the honest explanation for "why not lower," not "ran out of time."

### Stage 7 — Visualize (`src/viz/` + `src/spectral/`)
- `fourier_surface.py`: truncated 2D Fourier series representation of the DEM (Problem 7) —
  this is literally the data structure fed to the React Three Fiber mesh; re-tessellatable
  at any resolution without re-reading source imagery, and doubles as a clean low-pass
  filter / scan-line-artifact remover.
- `contours.py`: `skimage.measure.find_contours` — sub-pixel contour placement is linear
  interpolation along each grid edge (Problem 8), i.e. marching squares.
- Frontend R3F component displaces a plane mesh by the heightmap grid, draws contour lines
  at the levels returned by `/terrain/{crater_id}`.

### Stage 8 — Frontend (already scaffolded)
- Crater selector (limited DB, currently Tycho only) → `/match/{id}` → percentage-match
  card + keypoint overlay → `/terrain/{id}` → holographic R3F mesh.
- Supabase for scenes/jobs/matches/metrics tables + storage buckets; Upstash for job
  status + narration rate-limiting; Sentry breadcrumbs after each pipeline stage so a
  silent band-order or coordinate-transform bug surfaces as an event, not a bad number;
  Gemini for narrating the metrics table to a non-technical judge (narration only, never
  for generating the metrics themselves).

### Stage 9 — Training (descriptor only; everything else above is closed-form)
- Crop DEM → Lommel-Seeliger-rendered synthetic sun-angle pairs → triplet-loss Siamese CNN
  → validate by ranking accuracy `d(a,p) < d(a,n)` → export ONNX (opset 17, 1–5 MB).
- Train on Kaggle T4×2 (free) or a rented/owned RTX 6000; never upload raw scenes as a
  Kaggle Dataset, only DEM-patch crops and the final `.onnx`.

### Stage 10 — Deployment + smoke test
- Frontend → Vercel/Netlify, `VITE_*` env vars. Backend → small always-on VM or
  Render/Railway (avoid serverless — GPU-adjacent inference doesn't fit cold starts).
- Six-point smoke test before demo day: backend up → frontend up → scene row appears in
  Supabase → `/match` writes a Redis status key + a `matches` row → a forced malformed
  payload shows up in both Sentry projects → narration panel renders and the 21st rapid
  call in a minute is actually rate-limited.

---

## PART 2 — The Required Mathematics

Ten pieces of math, each mapped to exactly where it runs in the codebase.

### 1. Sub-pixel keypoint localization (Taylor expansion) — `matching/sift_refine.py`
The DoG response `D(x, y, σ)` has a discrete max at `x₀ = (x₀, y₀, σ₀)`. Second-order
Taylor expansion about `x₀`:

```
D(x₀ + Δx) ≈ D(x₀) + (∂D/∂x)ᵀ Δx + ½ Δxᵀ (∂²D/∂x²) Δx
```

Setting the derivative w.r.t. `Δx` to zero gives the sub-pixel offset:

```
x̂ = − (∂²D/∂x²)⁻¹ (∂D/∂x)
```

One 3×3 Hessian inversion per candidate keypoint, computed from finite differences of the
local pixel values. This is why reported match locations aren't locked to the integer grid.

### 2. Homography as a null-space problem — `verify/dlt_homography.py`
Given correspondences `(xᵢ, x'ᵢ)` in homogeneous coordinates related by `H`:
`x'ᵢ × H xᵢ = 0`. Each correspondence contributes 2 independent linear equations in the 9
unknowns `h = vec(H)`. Stacking `n` correspondences: `A h = 0`, `A ∈ ℝ^(2n×9)`.

- Noise-free, `n ≥ 4`: `h` spans a 1-D null space of `A` (why 4 points is the minimum —
  8 constraints for 8 DOF, since `H` is scale-free).
- With noise, `A` has full rank, so solve `ĥ = argmin_{‖h‖=1} ‖A h‖²`.
- SVD: `A = U Σ Vᵀ`. Then `‖A h‖² = ‖Σ Vᵀh‖² = Σⱼ σⱼ² yⱼ²` where `y = Vᵀh`, minimized
  (subject to `‖y‖=1`) by putting all weight on the smallest singular value — so `ĥ` is
  the **last column of V** (a Rayleigh-quotient minimization, not a heuristic).

### 3. RANSAC iteration count as a probability calculation — `verify/ransac.py`
Inlier fraction `w`, minimal sample size `s` (= 4 for a homography).
- `P(all s inliers in one sample) = wˢ`
- `P(one sample not all-inlier) = 1 − wˢ`
- After `k` independent samples: `P(none all-inlier) = (1 − wˢ)^k`
- For confidence `p`: solve `(1 − wˢ)^k ≤ 1 − p` →

```
k ≥ log(1 − p) / log(1 − wˢ)
```

Worked number for a shadow-degraded set: `s = 4, w = 0.3, p = 0.99` → `wˢ = 0.0081` →
`k ≥ log(0.01)/log(0.9919) ≈ 567` iterations. Log this derived `w` per scene from your
matcher's own inlier history instead of hard-coding `max_iters`.

### 4. Stereo triangulation as ray intersection — `geometry/triangulate.py`
Two camera centers `C₁, C₂`, unit ray directions `d₁, d₂`. Closest-approach point between
skew lines `r₁ = C₁ + t₁d₁`, `r₂ = C₂ + t₂d₂`: the connecting vector is ⊥ to both directions.
With `w = C₁ − C₂`:

```
[ d₁·d₁   −d₁·d₂ ] [t₁]   [ −w·d₁ ]
[ d₁·d₂   −d₂·d₂ ] [t₂] = [ −w·d₂ ]
```

Solve the 2×2 system directly; the recovered ground point is the midpoint of the segment
joining `r₁(t₁)` and `r₂(t₂)`. Used on TMC-2's fore/nadir/aft stereo triplet to build a DEM
patch when an external one is thin for the chosen scene.

### 5. Cramér–Rao bound on localization precision — evaluation write-up (Stage 6)
Noiseless 1-D scanline intensity `I(x)` near a feature, observed as `I(x) + η`,
`η ~ N(0, σ²)` i.i.d. per pixel, `N` samples. For any unbiased estimator `x̂`:

```
Var(x̂) ≥ σ² / Σₖ (∂I/∂x(xₖ))²
```

Derivation sketch: log-likelihood `ℓ(x) = −(1/2σ²) Σₖ(yₖ − I(xₖ−x))² + const`; Fisher
information `𝓘(x) = −E[∂²ℓ/∂x²] = (1/σ²) Σₖ(∂I/∂x)²`; Cramér–Rao gives `Var(x̂) ≥ 1/𝓘(x)`.
Interpretation: a sharp crater rim (large gradient) → tight bound, precise localization
possible; a flat regolith patch (gradient ≈ 0) → bound blows up, no algorithm can do
better. Cite this whenever measured RMSE approaches the image's own gradient-content
ceiling — a stronger, honest answer than "ran out of time."

### 6. Mutual information as a similarity statistic — `matching/mutual_information.py`
For random variables `A, B` with joint `p(a,b)` and marginals `p(a), p(b)`:

```
I(A;B) = Σ_{a,b} p(a,b) log[ p(a,b) / (p(a)p(b)) ]
```

Proof `I(A;B) ≥ 0`, equality iff independent: rewrite as
`−Σ p(a,b) log[p(a)p(b)/p(a,b)]`; Jensen's inequality on concave `log` gives
`≥ −log(Σ p(a)p(b)) = −log(1) = 0`. Equality holds iff `p(a)p(b)/p(a,b)` is constant
wherever `p(a,b) > 0`, which forces `p(a,b) = p(a)p(b)` everywhere. Used as the IIRS
cross-modal candidate branch: build the joint histogram of a patch pair's intensities (or
a chosen spectral-index projection), normalize, sum directly.

### 7. Terrain as a truncated 2D Fourier series — `spectral/fourier_surface.py`
Doubly-periodic `z(x,y)` on `[0,Lx]×[0,Ly]`:

```
z(x,y) = Σ_{m=−M}^{M} Σ_{n=−N}^{N} c_mn e^{i2π(mx/Lx + ny/Ly)}

c_mn = (1/LxLy) ∫∫ z(x,y) e^{−i2π(mx/Lx + ny/Ly)} dx dy
```

- Coefficients are computed **exactly** for sampled DEM data via `numpy.fft.fft2` — no
  fitting uncertainty, only the choice of how many `(M,N)` terms to keep.
- Truncating high-`|m|,|n|` terms is exactly a low-pass filter (they carry pixel-level
  noise, not macroscopic terrain shape).
- A periodic scan-line artifact of period `T` concentrates its energy in one `(m,n)` bin
  (or a narrow neighborhood, for period drift); zeroing that bin and inverse-transforming
  removes it without ever estimating its amplitude in the spatial domain.
- This coefficient grid is exactly what's fed to the R3F terrain mesh — evaluable at any
  render resolution without re-reading source imagery.

### 8. Sub-pixel contour placement (marching squares) — `viz/contours.py`
Unit grid-cell edge with corner heights `z₁ < z₂`, contour level `zₖ` in between. Linear
interpolation along the edge, `z(t) = z₁ + t(z₂ − z₁)`, `t ∈ [0,1]`:

```
t = (zₖ − z₁)/(z₂ − z₁),   P = P₁ + t(P₂ − P₁)
```

Same section-formula idea as dividing a segment in a given ratio, just with the ratio
determined by height. Apply to every sign-changing cell edge, connect per the cell's
corner sign-pattern → marching squares (`skimage.measure.find_contours` implements this;
you should still be able to derive it live if asked).

### 9. Collinearity — pixel-to-ground transform — `geometry/pixel_to_ground.py`
Camera at `C`, orientation `R` (camera→world), image-plane point `(u,v)`, focal length `f`:

```
λ (u, v, −f)ᵀ = Rᵀ(P − C),   λ > 0
```

Ray direction `d = R(u,v,−f)ᵀ`; ray: `P = C + s·d`. Substitute `P`'s z-coordinate into the
DEM surface constraint `P_z = h(P_x, P_y)` → one scalar equation in one unknown `s`
(root-find, or closed-form if `h` is locally planar). Once `s` is found, `P` is fully
determined. `R`, `C` come from orbit/attitude ephemeris metadata; `f` and the `(u,v)→ray`
mapping from sensor interior calibration (also in the PDS4 label); `h(x,y)` comes from the
DEM. **Nothing here is estimated from image content** — this is exactly why it's
closed-form geometry, not an open research question. Used both to reproject reference into
source frame, and to convert final matched keypoints into selenographic lat/lon.

### 10. Gaussian pyramid: blur-then-downsample, and why order matters — `matching/pyramid.py`
`Gσ` = convolution with a Gaussian of std `σ`; pyramid step:
`I_{l+1}(x,y) = (Gσ * I_l)(2x, 2y)`.

Downsampling by 2 halves the Nyquist frequency. Any content in `I_l` above the *new*
Nyquist limit that isn't removed first will **alias** — reappear as spurious low-frequency
content indistinguishable from real coarse structure (same mechanism as a wagon wheel
appearing to spin backwards under an undersampled frame rate). `Gσ` low-pass-filters that
content away *before* the irreversible information loss of downsampling. Get the order
backwards and coarse-to-fine correspondences (across the ~17–20:1 OHRC-to-TMC-2 scale
ratio) break silently, with no error thrown.

---

## Quick reference — problem → codebase location

| # | Concept | File |
|---|---|---|
| 1 | Taylor expansion, sub-pixel extrema | `matching/sift_refine.py` |
| 2 | SVD null-space, homography | `verify/dlt_homography.py` |
| 3 | Probability, RANSAC iteration count | `verify/ransac.py` |
| 4 | 3D vectors, ray intersection | `geometry/triangulate.py` |
| 5 | Fisher information, Cramér–Rao bound | evaluation write-up, Stage 6 |
| 6 | Jensen's inequality, mutual information | `matching/mutual_information.py` |
| 7 | 2D Fourier series | `spectral/fourier_surface.py` |
| 8 | Linear interpolation, contours | `viz/contours.py` |
| 9 | Linear algebra, collinearity | `geometry/pixel_to_ground.py` |
| 10 | Sampling theorem, aliasing | `matching/pyramid.py` |
