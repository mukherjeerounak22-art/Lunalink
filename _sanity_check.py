"""End-to-end sanity check for SIH26166 (run against a live localhost:8000).

Usage: python _sanity_check.py [BASE_URL]
       default BASE_URL = http://127.0.0.1:8000 — pass your deployed
       backend URL (e.g. https://sih26166-backend.onrender.com) to audit
       the Render deployment from anywhere.
"""
import io
import json
import os
import sys
import time
import zipfile

import httpx

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ROOT = os.path.dirname(os.path.abspath(__file__))


def timed(label, fn):
    t0 = time.time()
    try:
        out = fn()
        print("%s OK (%.1fs)" % (label, time.time() - t0))
        return out
    except Exception as exc:
        print("%s FAILED (%.1fs): %s" % (label, time.time() - t0, exc))
        return None


c = httpx.Client(timeout=600)

health = timed("GET /health", lambda: c.get(BASE + "/health").json())
print("  learned_model_loaded:", health and health["learned_model_loaded"])

# ---- 1. plain image upload -> scene with AUTO-SELECTED reference ----
demo = os.path.join(ROOT, "data", "demo_upload", "ch2_ohrc_real_crop_web.jpg")
r = timed("POST /analyze_upload (make_scene=true, real OHRC jpg)", lambda: c.post(
    BASE + "/analyze_upload",
    files={"file": ("ch2_ohrc_real_crop_web.jpg", open(demo, "rb"), "image/jpeg")},
    data={"sun_az": "270.8", "sun_el": "10", "make_scene": "true",
          "scene_name": "sanity_upload"}))
scene_id = None
if r is not None:
    j = r.json()
    scene_id = j.get("created_scene")
    meta = j.get("metadata", {})
    print("  scene:", scene_id, "| craters:", len(j.get("craters", [])),
          "| relief: %.1f-%.1f m" % (j["grid"]["zmin_m"], j["grid"]["zmax_m"]))
    print("  reference_source:", meta.get("reference_source", "?")[:110])

# ---- 2. match: ONNX branch + SIFT + RANSAC + Fourier-Mellin ----
if scene_id:
    m = timed("GET /match/" + scene_id, lambda: c.get(BASE + "/match/" + scene_id).json())
    if m:
        mb = m["method_breakdown"]
        fm = m.get("fourier_mellin", {})
        print("  match%%: %.2f | inliers: %s | RMSE: %s px" %
              (m["match_percentage"], m["inlier_count"], m["rmse_px"]))
        print("  SIFT candidates:", mb["sift_candidates"],
              "| ONNX learned candidates:", mb["learned_candidates"],
              "| model:", mb["learned_model_loaded"])
        print("  fourier-mellin: rot=%s deg, scale=%s, t=%s, lp-resp=%s" %
              (fm.get("rotation_deg"), fm.get("scale"),
               fm.get("translation_px"), fm.get("logpolar_response")))
        print("  ref image path:", m["reference_image"])
        print("  ref caption data:", (m.get("metadata", {}).get("reference_source") or "?")[:90])

# ---- 3. terrain: grid + contours + sun geometry ----
    t = timed("GET /terrain/" + scene_id, lambda: c.get(BASE + "/terrain/" + scene_id).json())
    if t:
        g = t["grid"]
        print("  grid %dx%d, extent %.0f m, relief %.1f-%.1f m, contour levels %d, segs %d" %
              (g["n"], g["n"], g["extent_m"], g["zmin_m"], g["zmax_m"],
               len(t["contours"]["levels_m"]),
               sum(len(s) for s in t["contours"]["segments"])))
        print("  sun:", t["metadata"]["sun"], "| craters:",
              t["metadata"].get("craters_detected"))

# ---- 4. narration (Gemini vs local fallback) ----
    n = timed("GET /narrate/" + scene_id, lambda: c.get(BASE + "/narrate/" + scene_id).json())
    if n:
        print("  narration source:", n["source"])
        print("  narration:", n["narration"][:220], "...")

# ---- 5. product upload: ISRO PDS4 XML+IMG pair ----
raw = os.path.join(ROOT, "data", "raw", "demo_tmc")
pair = [("files", ("ch2_tmc_demo_d_img_d18.xml", open(os.path.join(raw, "ch2_tmc_demo_d_img_d18.xml"), "rb"), "application/xml")),
        ("files", ("ch2_tmc_demo_d_img_d18.img", open(os.path.join(raw, "ch2_tmc_demo_d_img_d18.img"), "rb"), "application/octet-stream"))]
r = timed("POST /ingest_product_upload (PDS4 xml+img pair)", lambda: c.post(
    BASE + "/ingest_product_upload", files=pair, data={"scene_name": "sanity_tmc_product"}))
if r is not None:
    j = r.json()
    print("  scene:", j["created_scene"], "| craters:", len(j["craters"]),
          "| label:", j.get("label"))
    print("  reference:", (j["terrain"]["metadata"].get("reference_source") or "?")[:110])

# ---- 6. product upload: ZIP of the product directory ----
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    for fn in os.listdir(raw):
        z.write(os.path.join(raw, fn), "product/" + fn)
buf.seek(0)
r = timed("POST /ingest_product_upload (zip)", lambda: c.post(
    BASE + "/ingest_product_upload",
    files={"files": ("demo_product.zip", buf, "application/zip")},
    data={"scene_name": "sanity_tmc_zip"}))
if r is not None:
    j = r.json()
    print("  scene:", j["created_scene"], "| craters:", len(j["craters"]))

# ---- 7. registry + frontend served ----
reg = json.load(open(os.path.join(ROOT, "data", "processed", "registry.json")))
print("registry scenes:", list(reg.keys()))
html = c.get(BASE + "/").text
print("frontend served:", len(html), "bytes | refCap:", "refCap" in html,
      "| product upload wired:", "ingest_product_upload" in html)
