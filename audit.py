"""End-to-end audit: verifies EVERY frontend binding has real backend data
behind it and every integration actually works. Run from repo root.
Keys are read from backend/.env - never hard-coded."""
import json
import os
import re
import urllib.request

import httpx

BASE = "http://localhost:8000"

# load backend/.env
_env = {}
for _line in open(os.path.join("backend", ".env"), encoding="utf-8"):
    _line = _line.strip()
    if _line and "=" in _line and not _line.startswith("#"):
        _k, _v = _line.split("=", 1)
        _env[_k.strip()] = _v.strip()
SB = _env["SUPABASE_URL"]
SK = _env["SUPABASE_SERVICE_ROLE_KEY"]
UP = _env["UPSTASH_REDIS_REST_URL"]
UT = _env["UPSTASH_REDIS_REST_TOKEN"]
ok_count, fail = 0, []


def check(label, cond, detail=""):
    global ok_count
    if cond:
        ok_count += 1
        print("  PASS", label, detail)
    else:
        fail.append(label)
        print("  FAIL", label, detail)


def get(path, timeout=300):
    return json.load(urllib.request.urlopen(BASE + path, timeout=timeout))


print("=== 1. health + learned ONNX ===")
h = get("/health")
check("health ok", h["status"] == "ok")
check("learned ONNX loaded", h["learned_model_loaded"] is True)

print("=== 2. /craters feeds scene selector + metadata panel ===")
craters = get("/craters")
check("2 scenes", len(craters) == 2)
real = [c for c in craters if c["kind"] == "real"][0]
m = real["metadata"]
check("metadata fields for metaKv panel",
      all(k in m for k in ("product_id", "start_time_utc", "band",
                           "pixel_resolution_m", "spacecraft_altitude_km",
                           "sun", "dem_range_m", "provenance")))

print("=== 3. /match feeds MATCH window (all bound fields) ===")
match = get("/match/ohrc_20210401")
required = ["match_percentage", "inlier_count", "matches", "rmse_px",
            "n_keypoints_source", "n_keypoints_ref", "cache", "scene_id",
            "method_breakdown", "ransac", "source_image", "reference_image",
            "metadata", "evaluation_notes"]
check("all /match fields the UI binds", all(k in match for k in required),
      str([k for k in required if k not in match]))
check("matches have src/ref/inlier", all(
    set(mt) >= {"src", "ref", "inlier"} and len(mt["src"]) == 2
    for mt in match["matches"][:20]))
mb = match["method_breakdown"]
check("method_breakdown fields for card",
      all(k in mb for k in ("sift_candidates", "learned_model_loaded",
                            "learned_candidates")))
check("ransac derived k", match["ransac"]["derived_iterations_k"] >= 1)
check("homography 3x3", len(match["homography"]) == 3)


print("=== 4. images served (correspondence canvases) ===")
for img in (match["source_image"], match["reference_image"]):
    r = httpx.get(BASE + img, timeout=30)
    check("image %s" % img, r.status_code == 200 and len(r.content) > 100000)

print("=== 5. /terrain feeds TERRAIN 3D window ===")
t = get("/terrain/ohrc_20210401")
g = t["grid"]
check("grid n=192 heights", g["n"] == 192 and len(g["heights_m"]) == 192
      and len(g["heights_m"][0]) == 192)
check("contours levels+segments", len(t["contours"]["levels_m"]) == 8 and
      all(isinstance(s, list) for s in t["contours"]["segments"]))
check("cache layer active", t["cache"] in ("fresh", "redis", "memory"))

print("=== 6. /analyze_upload (UPLOAD window) ===")
import io
import numpy as np
from PIL import Image
y, x = np.mgrid[0:128, 0:128].astype("float32")
r = np.sqrt((x - 64) ** 2 + (y - 64) ** 2) / 40
img = np.clip(128 - 60 * np.clip(1 - r, 0, 1), 0, 255).astype("uint8")
buf = io.BytesIO()
Image.fromarray(img).save(buf, format="PNG")
body = ("--B\r\nContent-Disposition: form-data; name=\"sun_az\"\r\n\r\n315\r\n"
        "--B\r\nContent-Disposition: form-data; name=\"sun_el\"\r\n\r\n30\r\n"
        "--B\r\nContent-Disposition: form-data; name=\"file\"; "
        "filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n").encode() + \
    buf.getvalue() + b"\r\n--B--\r\n"
req = urllib.request.Request(BASE + "/analyze_upload", data=body,
    headers={"Content-Type": "multipart/form-data; boundary=B"})
up = json.load(urllib.request.urlopen(req, timeout=300))
check("upload returns terrain payload",
      up["grid"]["n"] == 192 and len(up["contours"]["segments"]) == 8)

print("=== 7. /narrate (NARRATION window) ===")
n = get("/narrate/tycho")
check("narration source valid", n["source"] in ("gemini", "local-template"))
check("narration non-empty with metrics", len(n["narration"]) > 100
      and "%" in n["narration"])


print("=== 8. Supabase persistence ===")
hdr = {"apikey": SK, "Authorization": "Bearer " + SK}
for table, min_rows in (("scenes", 1), ("jobs", 2), ("matches", 2), ("metrics", 2)):
    rows = httpx.get(f"{SB}/rest/v1/{table}?select=*&limit=20", headers=hdr,
                     timeout=15).json()
    check(f"supabase {table} has rows", isinstance(rows, list) and
          len(rows) >= min_rows, f"({len(rows)})")

print("=== 9. Redis cache-aside + sliding-window limiter ===")
r1 = httpx.get(f"{UP}/get/match%3Aohrc_20210401",
               headers={"Authorization": "Bearer " + UT}, timeout=10).json()
check("redis match cache key", r1.get("result") not in (None, ""))
check("redis value is our payload", "match_percentage" in (r1.get("result") or ""))
r2 = httpx.get(f"{UP}/zcard/narrate%3Atycho",
               headers={"Authorization": "Bearer " + UT}, timeout=10).json()
check("sliding-window counter active", (r2.get("result") or 0) > 0)

print("=== 10. Sentry ===")
r = httpx.get(BASE + "/debug/sentry-test", timeout=20)
check("sentry test event accepted", r.status_code == 200)

print("=== 11. frontend bindings: every $('id') exists in HTML ===")
html = open("frontend/index.html", encoding="utf-8").read()
ids_used = set(re.findall(r"\$\(\"([\w-]+)\"\)", html)) | \
    set(re.findall(r"getElementById\('([\w-]+)'\)", html))
ids_defined = set(re.findall(r'id="([\w-]+)"', html))
missing = ids_used - ids_defined
check("all JS-referenced ids exist in markup", not missing, str(missing))

print("=== 12. frontend buttons all wired to click listeners ===")
buttons = set(re.findall(r'<button[^>]*id="([\w-]+)"', html))
wired = set(re.findall(r'\$\("([\w-]+)"\)\.addEventListener\("click"', html))
tabs = {"tab-match", "tab-terrain", "tab-upload", "tab-narrate"}
unwired = (buttons - tabs) - wired
check("non-tab buttons wired", not unwired, str(unwired))
check("tab bar complete", tabs <= buttons)

print("=== 13. static assets + model on disk ===")
r = httpx.get(BASE + "/config.js", timeout=10)
check("config.js served with frontend DSN", "SENTRY_DSN_FRONTEND" in r.text and
      "ingest.us.sentry.io" in r.text)
import os
check("descriptor.onnx on disk", os.path.exists("backend/models/descriptor.onnx"))

print()
print("AUDIT RESULT: %d passed, %d failed" % (ok_count, len(fail)))
if fail:
    print("FAILURES:", fail)


