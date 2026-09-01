# SIH26166 — Lunar Scene Matching + 3D Terrain Hologram

Cross-illumination registration of a **real Chandrayaan-2 OHRC** scene with a
holographic 3D terrain view, per `SIH26166_Implementation_Plan_and_Mathematics.md`
and `SIH26166_Tomorrow_Plan_Kaggle_and_Keys.md`.

> **Docs:** [`DEMO_GUIDE.md`](DEMO_GUIDE.md) — how to run it, demo test
> images, 60-second judge demo. [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md) —
> judge Q&A, full system walkthrough, ONNX training story, and the
> mathematical formulation of every stage.


## Dataset (already walked through, Stage 0/1)
`data/UNZIPPED_DATA/ch2_ohr_ncp_20210401T2357376656_d_img_d18/`
- **Image**: 90,148 × 12,000 px 8-bit panchromatic, GSD 0.265 m/px, alt 104.22 km,
  acquired 2021-04-01T23:57:37Z, descending orbit, footprint 13.06°S–13.89°S /
  25.13°E–25.25°E
- **Label** (PDS4 XML): focal length 2080 mm, 5.2 µm pixels, TDI64, exposure 181.74 ms
- **Geometry CSV**: per-pixel lon/lat ground coordinates
- **Sun file (.spm)**: sun elevation 9.92° → incidence 80.08°, azimuth 270.8°
  (near-terminator imaging — exactly the shadow-robustness challenge)

## Run
```bash
cd backend
python preprocess.py          # one-time: reads the 1 GB .img, builds DEM + pairs
python -m uvicorn main:app --port 8000
# open http://localhost:8000
```

## Pipeline → endpoints
| Stage | Where | Endpoint |
|---|---|---|
| 1 ingest | `backend/preprocess.py` (PDS4 label + .spm + memmap .img) | — |
| 2 geometry/relief | shape-from-shading, linearized Lambert + FFT Poisson (sun vector from mission metadata) | `/terrain/{id}` |
| 3 photometric | CLAHE normalization before matching | — |
| 4 matching | SIFT (DoG + sub-pixel Taylor refinement), pyramid via octaves, learned-branch hook | `/match/{id}` |
| 5 verify | RANSAC + DLT homography (SVD null-space); k ≥ log(1−p)/log(1−wˢ) derived per scene | `/match/{id}` |
| 6 evaluate | inlier ratio = match %, RMSE, method_breakdown, Cramér–Rao note | `/match/{id}` |
| 7 visualize | truncated 2D Fourier low-pass + marching squares (sub-pixel contours) | `/terrain/{id}` |
| 8 frontend | `frontend/index.html` — Three.js hologram, contour lines, match card | `/` |

## Scenes (crater dropdown)
1. **CH-2 OHRC real scene** — real ISRO radiance; relief from shape-from-shading
   (non-metric, slope-calibrated, stated in metadata); reference = simulated
   second-pass product (gamma + radiance gradient + noise + rotation/scale
   homography). Measured demo values: ~99.5 % inlier ratio, RMSE ≈ 0.31 px.
2. **Tycho (synthetic stand-in)** — Kaggle-notebook DEM, Lommel–Seeliger pair at
   two sun geometries with regolith albedo texture. Measured: ~59 % inlier ratio,
   RMSE ≈ 0.27 px — the honest cross-illumination number.

## Honesty framing (per the plan docs)
- The relief for the real scene is a photometric approximation, **not** a metric
  DEM — the frontend metadata panel says so verbatim.
- Synthetic numbers are never mixed with real-scene numbers (separate scenes).
- Learned ONNX branch not trained yet → backend reports `learned_model_loaded:
  false` and runs the documented SIFT-only fallback.
- Train `descriptor.onnx` on Kaggle (notebook cells in the Tomorrow Plan doc),
  drop it in `backend/models/descriptor.onnx`, and the method breakdown lights up.

## Requirements
`pip install fastapi uvicorn numpy opencv-python scipy pillow`
Frontend needs internet once (Three.js CDN via import map).
