"""Bake the polar Level-6 OHRC scene from the Kaggle-hosted p1d product.

Pulls the calibrated product (xml + img + geometry csv) per-file via
kagglehub, assembles a staging dir, runs the full ingest (SFS, craters,
NASA/ISRO reference selection), names the scene, and cleans up."""
import os, sys, shutil, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.chdir(os.path.join(ROOT, "backend"))

import kfetch

SLUG = "rounakmukherjee22/ohrc-polar-p1d"
OWNER, DSLUG = SLUG.split("/", 1)
PID = "ch2_ohr_ncp_20211222T2023166276_d_img_d32"
DATE = "20211222"
SCENE_ID = "ohrc_polar_20211222"
PREFIX = ""          # p1d layout: members at the dataset root

STAGE = os.path.join(ROOT, "data", "staging", "p1d")
PARTS = [
    (f"data/calibrated/{DATE}/{PID}.img", f"data/calibrated/{DATE}/{PID}.img"),
    (f"data/calibrated/{DATE}/{PID}.xml", f"data/calibrated/{DATE}/{PID}.xml"),
    (f"geometry/calibrated/{DATE}/{PID.split('_d_')[0]}_g_grd_d32.csv",
     f"geometry/calibrated/{DATE}/{PID.split('_d_')[0]}_g_grd_d32.csv"),
    (f"geometry/calibrated/{DATE}/{PID.split('_d_')[0]}_g_grd_d32.xml",
     f"geometry/calibrated/{DATE}/{PID.split('_d_')[0]}_g_grd_d32.xml"),
]
for remote, rel in PARTS:
    dst = os.path.join(STAGE, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 1e3:
        print("have:", rel)
        continue
    print("fetching:", rel)
    ok = kfetch.download_dataset_file(OWNER, DSLUG, remote, dst)
    if not ok:
        if "geometry" in rel:
            print("   (optional - skipping)")
            if os.path.exists(dst):
                os.remove(dst)
            continue
        raise SystemExit("download failed: " + rel)
    print("   %.1f MB" % (os.path.getsize(dst) / 1e6))

import ingest
res = ingest.ingest_product_dir(STAGE, scene_id=SCENE_ID)
print("scene:", res["scene_id"], "| craters:", len(res.get("craters", [])))

# name it in the registry
reg_path = os.path.join(ROOT, "data", "processed", "registry.json")
reg = json.load(open(reg_path))
if SCENE_ID in reg:
    reg[SCENE_ID]["name"] = ("Level 6 - CH-2 OHRC polar scene "
                             "(-84.5S, 311.6E)")
    reg[SCENE_ID]["subtitle"] = ("polar OHRC %s - IIRS-consistent region"
                                 % PID)
    json.dump(reg, open(reg_path, "w"), indent=1)
    print("registry named:", reg[SCENE_ID]["name"])

meta = json.load(open(os.path.join(ROOT, "data", "processed", SCENE_ID,
                                   "metadata.json")))
print("center:", meta.get("source_footprint_center"))
print("refs:", {k: (meta.get(k) or {}).get("status")
                for k in ("tmc_reference", "iirs_reference")})
