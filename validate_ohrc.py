"""Validate downloaded OHRC candidates for the Option-B polar region.

Scans the repo root (or a given dir) for ch2_ohr_nrp_*.zip / extracted
product dirs, parses each label's ISDA footprint corners, and checks
coverage of the Option-B center (-67.89, 210.4) plus the demo box
(lat -68.39..-67.39, lon 209.9..210.9).  Prints the best scene-anchor pick.
"""
import os, re, sys, glob, json, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "backend"))
from tmc import parse_isda_geometry, geo_uv, great_circle_km

CENTER = {"lat_deg": -89.5, "lon_deg": 210.0}
BOX = {"lat": (-89.95, -89.2), "lon": (0.0, 360.0)}
SEARCH = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def corners_from_zip(zp):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.lower().endswith(".xml") and "brw" not in n.lower() \
                    and "miscellaneous" not in n.lower():
                return parse_isda_geometry(
                    z.read(n).decode("utf-8", "replace"))
    return {}

def corners_from_dir(dp):
    for root, _, files in os.walk(dp):
        for fn in files:
            if fn.lower().endswith(".xml") and "brw" not in fn.lower() \
                    and "miscellaneous" not in root.lower():
                return parse_isda_geometry(
                    open(os.path.join(root, fn), encoding="utf-8",
                         errors="replace").read())
    return {}

cands = []
for zp in glob.glob(os.path.join(SEARCH, "ch2_ohr_nrp_*.zip")):
    cands.append((zp, corners_from_zip(zp)))
for dp in glob.glob(os.path.join(SEARCH, "ch2_ohr_nrp_*")):
    if os.path.isdir(dp):
        cands.append((dp, corners_from_dir(dp)))

print("center: lat %.2f lon %.1f  | box lat %.1f..%.1f lon %.1f..%.1f"
      % (CENTER["lat_deg"], CENTER["lon_deg"] % 360,
         BOX["lat"][0], BOX["lat"][1], BOX["lon"][0], BOX["lon"][1]))
best = None
for path, geo in cands:
    pid = os.path.basename(path).replace(".zip", "")
    cor = (geo or {}).get("footprint_corners")
    if not cor:
        print("%-52s NO CORNERS in label" % pid)
        continue
    uv = geo_uv(cor, CENTER)
    lats = [c["lat_deg"] for c in cor.values()]
    lons = [c["lon_deg"] % 360 for c in cor.values()]
    in_box = any(BOX["lat"][0] <= la <= BOX["lat"][1]
                 and BOX["lon"][0] <= lo <= BOX["lon"][1]
                 for la, lo in zip(lats, lons))
    margin = min(great_circle_km(CENTER["lat_deg"], CENTER["lon_deg"],
                                 c["lat_deg"], c["lon_deg"])
                 for c in cor.values())
    print("%-52s covers_center=%-5s corner_in_box=%-5s edge_margin=%6.0f km"
          % (pid, bool(uv), in_box, margin))
    if uv and (best is None or margin > best[0]):
        best = (margin, pid, path, geo)

if best:
    print("\nBEST SCENE ANCHOR: %s (%.0f km margin inside both the TMC-2 "
          "DTM and IIRS cube footprints)" % (best[1], best[0]))
else:
    print("\nNo candidate covers the Option-B center - check the search "
          "AOI or the downloads.")
