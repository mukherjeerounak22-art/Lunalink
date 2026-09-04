"""Full Kaggle account inventory + triple-coverage verification."""
import os, sys, json, glob
sys.path.insert(0, r"c:\Users\user\Downloads\SIH\backend")
from tmc import parse_isda_geometry, geo_uv

ROOT = r"c:\Users\user\Downloads\SIH"
CFG = json.load(open(os.path.join(ROOT, "kaggle.json")))
AUTH = (CFG["username"], CFG["key"])
USER = CFG["username"]
ENV = dict(os.environ, KAGGLE_CONFIG_DIR=os.path.join(ROOT))

import subprocess
def cli(args):
    return subprocess.run([sys.executable, "-m", "kaggle"] + args,
                          capture_output=True, text=True, env=ENV).stdout

def list_all(slug):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    names, token = [], None
    for _ in range(40):
        res = api.dataset_list_files(slug, page_token=token or None,
                                     page_size=100)
        for f in (getattr(res, "files", None) or []):
            n = getattr(f, "name", None) or getattr(f, "path", None)
            if n:
                names.append(n)
        token = getattr(res, "next_page_token", None)
        if not token:
            break
    return names

def label_corners(slug, path):
    import requests
    url = (f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
           f"?fileName={requests.utils.quote(path)}")
    r = requests.get(url, auth=AUTH, stream=True, timeout=60)
    if r.status_code != 200:
        return None
    return parse_isda_geometry(
        r.content.decode("utf-8", "replace")).get("footprint_corners")

# the two verified-consistent footprints
lib = json.load(open(os.path.join(ROOT, "data", "reference", "tmc",
                                  "_library.json")))
TMC = (lib["products"]["ch2_tmc_ndn_20231109T2148028796_d_dtm_d18"]
       .get("geometry") or {}).get("footprint_corners")
import zipfile, requests
IIRS_LABEL = ("ch2_iir_nci_20221227T0748212038_d_img_d32/data/calibrated/"
              "20221227/ch2_iir_nci_20221227T0748212038_d_img_d32.xml")
r = requests.get(
    f"https://www.kaggle.com/api/v1/datasets/download/{USER}/"
    f"ch2-iirs-tmc2-data?fileName={requests.utils.quote(IIRS_LABEL)}",
    auth=AUTH, timeout=60)
IIRS = parse_isda_geometry(
    r.content.decode("utf-8", "replace")).get("footprint_corners")
print("TMC-2 DTM 20231109T2148 covers (-67.89, 210.4E):",
      bool(geo_uv(TMC, {"lat_deg": -67.89, "lon_deg": 210.4})))
print("IIRS cube 20221227 covers (-67.89, 210.4E):",
      bool(geo_uv(IIRS, {"lat_deg": -67.89, "lon_deg": 210.4})))
print()

# inventory: all datasets, find OHRC products + DTM coverage
print("=== datasets with OHRC products ===")
seen_products = {}
for slug in (f"{USER}/ohrc-polar-part2", f"{USER}/ohrc-polar-p1a",
             f"{USER}/ohrc-polar-p1b", f"{USER}/ohrc-polar-p1d",
             f"{USER}/ohrc-polar-p1c-v2", f"{USER}/final-kaggle-dataset"):
    try:
        names = list_all(slug)
    except Exception as e:
        print(slug, "list failed:", str(e)[:80]); continue
    prods = {}
    for n in names:
        base = n.split("/")[0]
        if base.startswith("ch2_ohr_") and ("/data/" in ("/" + n)):
            prods.setdefault(base, n.replace(base + "/", ""))
    if not prods:
        print(slug, "-> no OHRC product dirs"); continue
    print(slug)
    for base, datapath in prods.items():
        if base in seen_products:
            print("   dup:", base); continue
        seen_products[base] = (slug, datapath)
        xpath = datapath.replace(".img", ".xml")
        cor = label_corners(slug, xpath)
        if not cor:
            print("   %-52s no corners (label fetch failed)" % base)
            continue
        in_t = bool(geo_uv(TMC, {"lat_deg": sum(c['lat_deg'] for c in cor.values())/4,
                                        "lon_deg": sum(c['lon_deg'] % 360 for c in cor.values())/4}))
        in_i = bool(geo_uv(IIRS, {"lat_deg": sum(c['lat_deg'] for c in cor.values())/4,
                                        "lon_deg": sum(c['lon_deg'] % 360 for c in cor.values())/4}))
        lats = [c["lat_deg"] for c in cor.values()]
        lons = [c["lon_deg"] % 360 for c in cor.values()]
        print("   %-52s lat %8.2f..%8.2f lon %6.1f..%6.1f  TMC=%s IIRS=%s"
              % (base, min(lats), max(lats), min(lons), max(lons),
                 in_t, in_i))

print()
print("=== TMC-2 DTM coverage of the zone (from final-kaggle-dataset + parts) ===")
for slug in (f"{USER}/final-kaggle-dataset",
             f"{USER}/ch2-tmc2-derived-part-3"):
    try:
        names = list_all(slug)
    except Exception as e:
        print(slug, "list failed"); continue
    dtms = {n.split("/")[0] for n in names
            if "_d_dtm_" in n and n.endswith(".tif")}
    print(slug, "-> DTM products:", sorted(dtms)[:6])
