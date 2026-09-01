"""End-to-end test of dynamic ingestion: build a synthetic PDS4 product,
ingest it via /ingest_product, verify it becomes a full matchable scene."""
import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(ROOT, "data", "raw", "demo_tmc")
os.makedirs(PROD, exist_ok=True)

# synthetic 3000x2200 int16 "TMC-like" image with craters
y, x = np.mgrid[0:2200, 0:3000].astype(np.float32)
img = 128 + 10 * np.sin(x / 90.0) * np.cos(y / 70.0)
rng = np.random.default_rng(5)
for cx, cy, r, depth in [(600, 700, 120, 45), (1700, 1500, 200, 60),
                         (2400, 400, 90, 35)]:
    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    img -= depth * np.clip(1 - (rr / r) ** 2, 0, 1)
    img += depth * 0.4 * np.exp(-((rr - r) ** 2) / (2 * (r * 0.08) ** 2))
img += rng.normal(0, 4, img.shape)
img16 = np.clip(img, 0, 255).astype("<i2")
img16.tofile(os.path.join(PROD, "ch2_tmc_demo_d_img_d18.img"))

xml = """<?xml version="1.0"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
 <File_Area_Observational>
  <File><file_name>ch2_tmc_demo_d_img_d18.img</file_name></File>
  <Header><offset unit="byte">0</offset></Header>
  <Array_2D_Image>
   <offset unit="byte">0</offset>
   <Element_Array><data_type>SignedLSB2</data_type></Element_Array>
   <Axis_Array><axis_name>Line</axis_name><elements>2200</elements></Axis_Array>
   <Axis_Array><axis_name>Sample</axis_name><elements>3000</elements></Axis_Array>
  </Array_2D_Image>
 </File_Area_Observational>
</Product_Observational>
"""
open(os.path.join(PROD, "ch2_tmc_demo_d_img_d18.xml"), "w").write(xml)
print("synthetic PDS4 product written:", PROD)

BASE = "http://localhost:8000"

# 1. ingest
req = urllib.request.Request(
    BASE + "/ingest_product?path=demo_tmc&scene_id=demo_tmc&sun_az=250&sun_el=15",
    data=b"")  # POST
res = json.load(urllib.request.urlopen(req, timeout=300))
print("ingested:", res["scene_id"], "| DEM:", res["dem_range_m"],
      "| craters:", len(res["craters"]))

# 2. scene appears in /craters
craters = json.load(urllib.request.urlopen(BASE + "/craters", timeout=30))
print("/craters scenes:", [c["id"] for c in craters])

# 3. full match works on the ingested scene
m = json.load(urllib.request.urlopen(BASE + "/match/demo_tmc", timeout=300))
print("match on ingested scene: %.1f%% | inliers %d/%d | craters %d"
      % (m["match_percentage"], m["inlier_count"], len(m["matches"]),
         len(m.get("craters", []))))

# 4. terrain works
t = json.load(urllib.request.urlopen(BASE + "/terrain/demo_tmc", timeout=120))
print("terrain: %.1f-%.1f m, %d contour levels"
      % (t["grid"]["zmin_m"], t["grid"]["zmax_m"],
         len(t["contours"]["levels_m"])))

# 5. upload with make_scene=true
buf = io.BytesIO()
Image.fromarray(np.clip(img[500:1620, 900:2020], 0, 255).astype(np.uint8)
                ).save(buf, format="PNG")
boundary = "B"
body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"sun_az\"\r\n\r\n250\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"sun_el\"\r\n\r\n15\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"make_scene\"\r\n\r\ntrue\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"scene_name\"\r\n\r\ncourt upload\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"patch.png\"\r\nContent-Type: image/png\r\n\r\n").encode() + \
    buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
req = urllib.request.Request(BASE + "/analyze_upload", data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
up = json.load(urllib.request.urlopen(req, timeout=300))
print("upload scene created:", up.get("created_scene"),
      "| craters:", len(up.get("craters", [])))
m2 = json.load(urllib.request.urlopen(
    BASE + "/match/" + up["created_scene"], timeout=300))
print("match on uploaded scene: %.1f%%" % m2["match_percentage"])
print("ALL DYNAMIC-INGESTION TESTS PASSED")
