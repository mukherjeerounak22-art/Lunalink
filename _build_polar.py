"""One-shot polar Level-6 build: DTM extraction, METRIC window, MINERALS
window (kfetch), NASA matcher — with delete-after-extract cleanup.

Usage: python _build_polar.py <step>   steps: dtm | metric | minerals | nasa
"""
import os, sys, json, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.chdir(os.path.join(ROOT, "backend"))

PID = "ch2_tmc_ndn_20231024T0832339656_d_dtm_d18"
TIF_ZIP = r"C:\Users\user\Downloads\ch2_tmc_ndn_20231024T0832339656_d_dtm_d18.zip"
SCENE = "ohrc_polar_20211222"
SCENE_DIR = os.path.join(ROOT, "data", "processed", SCENE)
TIF = os.path.join(ROOT, "data", "reference", "tmc", PID, "dtm.tif")
QUB = os.path.join(ROOT, "data", "reference", "iirs",
                   "ch2_iir_nci_20221227T0748212038_d_img_d32", "cube.qub")

step = sys.argv[1] if len(sys.argv) > 1 else ""

if step == "dtm":
    import zipfile
    os.makedirs(os.path.dirname(TIF), exist_ok=True)
    with zipfile.ZipFile(TIF_ZIP) as z:
        member = next(n for n in z.namelist()
                      if n.lower().endswith(".tif"))
        print("extracting", member, "(7.24 GB — a couple of minutes)")
        with z.open(member) as src, open(TIF + ".part", "wb") as dst:
            done = 0
            while True:
                chunk = src.read(64 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                done += len(chunk)
                print("  %.1f GB" % (done / 1e9))
    os.replace(TIF + ".part", TIF)
    print("dtm.tif ready:", "%.1f GB" % (os.path.getsize(TIF) / 1e9))

elif step == "metric":
    import layers, json
    meta = json.load(open(os.path.join(SCENE_DIR, "metadata.json")))
    lib = json.load(open(os.path.join(ROOT, "data", "reference", "tmc",
                                      "_library.json")))
    corners = (lib["products"][PID].get("geometry") or {}).get("footprint_corners")
    prod = {"product_id": PID, "product_kind": "DTM (derived)",
            "geometry": {"footprint_corners": corners}}
    g, m = layers.tmc2_metric_dem(prod, center=meta["source_footprint_center"],
                                  extent_m=(meta["analysis_grid"]["cell_meters"]
                                            * meta["analysis_grid"]["n"]))
    print("metric:", "OK" if g is not None else m.get("error"))
    if g is not None:
        print("  window:", (m.get("geographic_window") or {}).get("pixel_window"),
              "| relief:", float(g.max()))

elif step == "minerals":
    import layers, json, requests, re
    from tmc import parse_isda_geometry
    meta = json.load(open(os.path.join(SCENE_DIR, "metadata.json")))
    IPID = "ch2_iir_nci_20221227T0748212038_d_img_d32"
    # IIRS corners from the Kaggle-hosted label (authoritative)
    cfg = json.load(open(os.path.join(ROOT, "kaggle.json")))
    r = requests.get(
        "https://www.kaggle.com/api/v1/datasets/download/"
        "rounakmukherjee22/ch2-iirs-tmc2-data?fileName="
        + requests.utils.quote(
            "ch2_iir_nci_20221227T0748212038_d_img_d32/browse/calibrated/"
            "20221227/ch2_iir_nci_20221227T0748212038_b_brw_d32.xml"),
        auth=(cfg["username"], cfg["key"]), timeout=60)
    icor = parse_isda_geometry(
        r.content.decode("utf-8", "replace")).get("footprint_corners")
    # cube axes from the data label
    xn = ("ch2_iir_nci_20221227T0748212038_d_img_d32/data/calibrated/"
          "20221227/ch2_iir_nci_20221227T0748212038_d_img_d32.xml")
    r2 = requests.get(
        "https://www.kaggle.com/api/v1/datasets/download/"
        "rounakmukherjee22/ch2-iirs-tmc2-data?fileName="
        + requests.utils.quote(xn), auth=(cfg["username"], cfg["key"]),
        timeout=60)
    xtxt = r2.content.decode("utf-8", "replace")

    def axis(name):
        m = re.search(r"<axis_name>%s</axis_name>\s*<elements>(\d+)"
                      % name, xtxt, re.I)
        return int(m.group(1)) if m else 0

    bands, glines, gsamples = (axis("Band") or axis("BAND"),
                               axis("Line") or axis("LINE"),
                               axis("Sample") or axis("SAMPLE"))
    print("cube axes: bands", bands, "lines", glines, "samples", gsamples)
    iprod = {"product_id": IPID, "product_kind": "spectral cube",
             "bands": bands, "lines": glines, "samples": gsamples,
             "geometry": {"footprint_corners": icor}}
    # persist a geometry.hdr so future runs know the shape without refetch
    cdir = os.path.join(ROOT, "data", "reference", "iirs", IPID)
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, "geometry.hdr"), "w") as f:
        f.write("bands = %d\nlines = %d\nsamples = %d\n"
                % (bands, glines, gsamples))
    classes, m = layers.iirs_minerals(iprod, center=meta["source_footprint_center"],
                                      extent_m=(meta["analysis_grid"]["cell_meters"]
                                                * meta["analysis_grid"]["n"]),
                                      force=True)
    print("minerals:", "OK" if classes is not None else m.get("error"))
    if classes is not None:
        print("  window:", (m.get("geographic_window") or {}).get("pixel_window"),
              "| classes present:", sorted(set(classes.ravel().tolist())))

elif step == "nasa":
    import ingest, cv2
    src = cv2.imread(os.path.join(SCENE_DIR, "source.png"),
                     cv2.IMREAD_GRAYSCALE)
    meta = json.load(open(os.path.join(SCENE_DIR, "metadata.json")))
    ref, m = ingest.auto_select_reference(
        src, max_candidates=None)
    print("nasa:", "SELECTED" if ref is not None else "SIMULATED",
          "|", (m or {}).get("product_id", ""),
          "| post-ncc:", (m or {}).get("translation", {}).get("ncc"),
          "| sift:", (m or {}).get("sift_refined"),
          (m or {}).get("inliers"))
    if (m or {}).get("note"):
        print("  note:", m["note"][:120])

elif step == "cleanup":
    import glob
    removed = 0
    for p in (TIF, QUB):
        if os.path.exists(p):
            os.remove(p); removed += 1
    if os.path.exists(TIF_ZIP):
        os.remove(TIF_ZIP); removed += 1
    print("removed:", removed, "big files (caches keep the layers alive)")

else:
    print("unknown step")
