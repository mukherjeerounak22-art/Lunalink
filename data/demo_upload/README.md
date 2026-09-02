# DEMO TEST IMAGES — SIH26166

Ready-to-upload files for the 03 UPLOAD window (dropzone accepts
`image/*`, `.zip`, `.img`, `.xml`). This folder IS committed to the repo
(`.gitignore` has a dedicated exception) so every teammate and the demo
machine get the exact same test images — clone and demo.

## Which file to use when

| File | Use it for |
|---|---|
| `level3_ch2_ohrc_real_crop_web.jpg` (346 KB) | **The main demo upload.** Real ISRO Chandrayaan-2 OHRC crop. Fast to drop in; overlaps an LRO NAC strip, so the reference is auto-selected from the real NASA library. |
| `level3_ch2_ohrc_real_crop_1024.png` | Same scene, lossless PNG — use if a judge asks about compression artifacts. |
| `level2_synthetic_craters_feature_rich.png` | Clean synthetic crater field — obvious craters make the shape-from-shading + 3D mesh visually undeniable. Great fallback if you want a crisp number. |
| `level2_synthetic_craters_256_fast.jpg` (21 KB) | Fastest possible live upload if the machine is slow. |
| `negative_control_low_feature.png` | Honest negative control: flat, low-contrast terrain. Upload it to show the pipeline does NOT invent relief when there is no signal (judges love this). |
| `level1_demo_tmc_product_crop_1024.png` | Image from the demo PDS4 product, for the "product → image" story. |
| `product_pds4_pair/` + `product_pds4_demo.zip` | Demo of the DIRECT PRODUCT UPLOAD: select both `ch2_tmc_demo_d_img_d18.xml` + `.img` together (or drop the zip). Backend parses the PDS4 label, ingests, and creates a full matchable scene automatically. |
| `reference_AUTO_selected_LROC_NAC.png` | NOT for upload — this is the auto-selected real NASA LRO NAC reference, for showing on a slide next to the source. |

## 60-second demo flow

1. Laptop backend: run `start_demo_tunnel.ps1` → open the printed
   `https://<vercel>/?api=<tunnel>` URL → badge green (or Render/localhost
   fallbacks — see `DEMO_INSTRUCTIONS.md` §0).
2. 01 MATCH: scene `CH-2 OHRC real scene` → RUN MATCH → keypoints,
   homography, RANSAC k, match% (real cross-mission number vs the
   auto-selected NASA reference).
3. 03 UPLOAD: drop `level3_ch2_ohrc_real_crop_web.jpg` → relief shading appears →
   SEND TO TERRAIN 3D (real polygon mesh) → CREATE MATCHABLE SCENE →
   scene appears in 01 MATCH with the auto-selected reference caption.
4. 02 TERRAIN stays interactive: orbit, inspect craters + contours.
5. 04 NARRATE: Gemini narrates the metrics in judge-friendly words
   (falls back to the local template automatically if quota runs out).

Regenerate the synthetic files: rerun the generator snippet in
`_sanity_check.py` sessions or ask the team — inputs are
`backend/preprocess.py::make_crater_dem` + `render_shaded`.
