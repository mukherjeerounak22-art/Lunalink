# DEMO GUIDE — SIH26166 (Lunalink)

How to run, what to upload, and the 60-second judge demo.

## Run it — two ways

**Primary (demo day): the Vercel link.**
The Vercel deployment is the frontend; it talks to the backend deployed on
Render (free tier) via `frontend/config.js:window.API_BASE`. One-time
setup (~20 min, Blueprint `render.yaml` included) is in
[`DEMO_INSTRUCTIONS.md` §14](DEMO_INSTRUCTIONS.md). Pre-warm the backend
(`https://<backend>.onrender.com/health` in a tab) ~5 min before demoing —
the free tier sleeps after ~15 min idle.

**Fallback (works fully offline): run everything locally.**

```bash
cd backend
uvicorn main:app --port 8000     # serves BOTH the API and the frontend
# open http://127.0.0.1:8000
```

Optional keys (all graceful no-ops without them) live in `backend/.env`
(never committed): Supabase, Upstash Redis, Sentry, Gemini. See
`backend/.env.example`.

## Demo assets — `data/demo_upload/` (committed to the repo)

The curated test-image set lives in the repo (gitignore exception), so a
fresh clone is demo-ready:

| File | Use it for |
|---|---|
| `ch2_ohrc_real_crop_web.jpg` | **Main demo upload** — real ISRO Chandrayaan-2 OHRC crop; overlaps an LRO NAC strip, so the reference is auto-selected from the real NASA library. |
| `ch2_ohrc_real_crop_1024.png` | Same scene, lossless PNG (compression-artifact questions). |
| `synthetic_craters_feature_rich.png` | Synthetic crater field — obvious craters make shape-from-shading + the 3-D mesh visually undeniable. |
| `synthetic_craters_256_fast.jpg` | 21 KB fast upload for a slow demo laptop. |
| `negative_control_low_feature.png` | Honest negative control: flat, low-contrast terrain — the pipeline does NOT invent relief when there is no signal. |
| `demo_tmc_product_crop_1024.png` | Image from the demo PDS4 product. |
| `product_pds4_pair/` + `product_pds4_demo.zip` | Direct product upload: select the `.xml` + `.img` pair together (or drop the zip) — the backend parses the PDS4 label and builds a full matchable scene automatically. |
| `reference_AUTO_selected_LROC_NAC.png` | Slide asset — the auto-selected real NASA LRO NAC reference. |

## 60-second demo flow

1. Open the **Vercel link** (backend pre-warmed — see `DEMO_INSTRUCTIONS.md`
   §0/§14) → health badge turns green. Local fallback: `http://127.0.0.1:8000`.
2. **01 MATCH** — scene *CH-2 OHRC real scene* → RUN MATCH → keypoints,
   homography, derived RANSAC budget, match % (the honest cross-mission
   number vs the auto-selected NASA reference), Fourier-Mellin row.
3. **03 UPLOAD** — drop `ch2_ohrc_real_crop_web.jpg` → reconstructed relief
   shading → SEND TO TERRAIN 3D (real polygon mesh) → CREATE MATCHABLE
   SCENE → the new scene appears in 01 MATCH with the auto-selected
   reference caption.
4. **02 TERRAIN** — orbit the mesh; craters, contours, relief readout.
5. **04 NARRATE** — Gemini narrates the metrics (local-template fallback
   is automatic if the quota is exhausted).

## Pre-demo sanity check

```bash
python _sanity_check.py                        # against a running localhost:8000
python _sanity_check.py https://<backend>.onrender.com   # against the Render deployment
```

(Also available against the deployed backend — pass its URL as the first
argument.) Covers: health + ONNX load, image upload → scene → match →
terrain → narration, PDS4 pair + ZIP product uploads, registry and frontend
checks. Test scenes are auto-removable from `data/processed/registry.json`.

See `PRESENTATION_GUIDE.md` for the full architecture walkthrough,
the ONNX training story, and the mathematical formulation.
