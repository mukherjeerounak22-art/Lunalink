# DEMO GUIDE — SIH26166 (Lunalink)

How to run, what to upload, and the 60-second judge demo.

## Run it — three ways

**Primary (demo day): Vercel frontend + backend on your laptop via tunnel.**
Run `start_demo_tunnel.ps1` (repo root) — it boots the backend, opens a
free cloudflared HTTPS tunnel, and prints the URL:
`https://<vercel>/?api=<tunnel>`. No cold starts; the full NASA LRO
reference library stays live for fresh uploads.

**Alternative (always-on): Vercel frontend + Render backend.**
`render.yaml` Blueprint included; setup in
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

Full path on the demo laptop: `c:\Users\user\Downloads\SIH\data\demo_upload\`
— open it in File Explorer and drag files into the 03 UPLOAD dropzone.
If the page ever hangs on `connecting…`, fully restart the browser (it
cached a failed DNS lookup) and reopen — all requests hard-time-out into a
retry state. The curated test-image set lives in the repo (gitignore
exception), so a fresh clone is demo-ready:

| File | Use it for |
|---|---|
| `level3_ch2_ohrc_real_crop_web.jpg` | **Main demo upload** — real ISRO Chandrayaan-2 OHRC crop; overlaps an LRO NAC strip, so the reference is auto-selected from the real NASA library. |
| `level3_ch2_ohrc_real_crop_1024.png` | Same scene, lossless PNG (compression-artifact questions). |
| `level2_synthetic_craters_feature_rich.png` | Synthetic crater field — obvious craters make shape-from-shading + the 3-D mesh visually undeniable. |
| `level2_synthetic_craters_256_fast.jpg` | 21 KB fast upload for a slow demo laptop. |
| `negative_control_low_feature.png` | Honest negative control: flat, low-contrast terrain — the pipeline does NOT invent relief when there is no signal. |
| `level1_demo_tmc_product_crop_1024.png` | Image from the demo PDS4 product (the Level-1 scene). |
| `level4_tmc2_metric_demo.png` | **Level 4** — TMC-2 DTM scene: the 02 TERRAIN layer switcher's METRIC (measured stereo heights) and SFS−METRIC validation layers are live here. |
| `level5_iirs_minerals_demo.png` | **Level 5** — IIRS mineral scene: the MINERALS layer (256-band cube band-depth classes + legend) is live here. |
| `product_pds4_pair/` + `product_pds4_demo.zip` | Direct product upload: select the `.xml` + `.img` pair together (or drop the zip) — the backend parses the PDS4 label and builds a full matchable scene automatically. |
| `reference_AUTO_selected_LROC_NAC.png` | Slide asset — the auto-selected real NASA LRO NAC reference. |

## 60-second demo flow

1. Open the **Vercel link** (backend pre-warmed — see `DEMO_INSTRUCTIONS.md`
   §0/§14) → health badge turns green. Local fallback: `http://127.0.0.1:8000`.
2. **01 MATCH** — scene *CH-2 OHRC real scene* → RUN MATCH → keypoints,
   homography, derived RANSAC budget, match % (the honest cross-mission
   number vs the auto-selected NASA reference), Fourier-Mellin row.
3. **03 UPLOAD** — drop `level3_ch2_ohrc_real_crop_web.jpg` → reconstructed relief
   shading → SEND TO TERRAIN 3D (real polygon mesh) → CREATE MATCHABLE
   SCENE → the new scene appears in 01 MATCH with the auto-selected
   reference caption.
4. **02 TERRAIN** — orbit the mesh; craters, contours, relief readout. The
   readout carries the scene's match % — every scene switch rebuilds it.
5. **04 NARRATE** — Gemini narrates the metrics (local-template fallback
   is automatic if the quota is exhausted); **🗣️ READ ALOUD** speaks it.
6. **Scene ladder** — the dropdown is leveled: Level 1 CH-2 TMC demo
   product (~59% match, easy) → Level 2 Tycho synthetic (~32%, medium) →
   Level 3 CH-2 OHRC real scene (~2.5% vs the auto-selected NASA strip,
   the honest cross-mission number) → **Level 4 TMC-2 metric DEM** (the
   METRIC + SFS−METRIC validation layers) → **Level 5 IIRS minerals**
   (the MINERALS layer with legend).

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
