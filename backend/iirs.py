"""IIRS spectral product library - the third leg of the ISRO reference
routes (OHRC source / TMC-TMC2 DTM / IIRS hyperspectral).

IIRS data is NOT yet on disk; this module is the plug-in point.  When a
PRADAN IIRS product arrives (extracted under data/raw/iirs/ or as
data/UNZIPPED_DATA/ch2_iir_*/), it is indexed here automatically: PDS4
spectral-cube label (Band/Line/Sample axes + ISDA footprint + sun), a
grayscale projection (browse PNG when present, else the middle band of the
memmapped cube), and the same select/build reference API as lroc.py and
tmc.py.  Until then every function degrades honestly to 'no data yet'.
"""
import json
import os
import re
import zipfile

import cv2
import numpy as np

from tmc import parse_isda_geometry, footprint_distance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IIRS_CACHE = os.path.join(ROOT, "data", "reference", "iirs")
IIRS_DIRS = [
    os.path.join(ROOT, "data", "raw", "iirs"),
    os.path.join(ROOT, "data", "UNZIPPED_DATA"),
]
# IIRS product zips are picked up straight from the repo root (the PRADAN
# download lands there) and from data/raw/iirs/ - so the library
# auto-populates the moment a download completes, zero manual steps
_PRODUCT_ZIP_GLOBS = [
    os.path.join(ROOT, "ch2_iir_*.zip"),
    os.path.join(ROOT, "data", "raw", "iirs", "*.zip"),
]
LIB_JSON = os.path.join(IIRS_CACHE, "_library.json")

_DTYPE = {"SignedLSB2": "<i2", "UnsignedLSB2": "<u2",
          "UnsignedMSB2": ">u2", "UnsignedByte": "u1",
          "SignedMSB2": ">i2"}


def _find_product_dirs():
    """Any directory carrying an IIRS label (ch2_iir_*) + raster/browse."""
    out = []
    for base in IIRS_DIRS:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            if "iir" not in os.path.basename(root).lower() and \
                    not any("iir" in f.lower() for f in files):
                dirs[:] = [d for d in dirs
                           if "iir" in d.lower()]      # prune non-IIRS trees
                continue
            xn = next((f for f in files
                       if f.lower().startswith("ch2_iir")
                       and f.lower().endswith(".xml")), None)
            if xn:
                out.append(root)
    return out


def _extract_iirs_zip(zf, src_note):
    """Index one IIRS product ZIP: cache-extract the small label + browse
    PNG into data/reference/iirs/<pid>/ and parse the geometry (the big
    spectral raster stays inside the zip).  Returns the entry or None."""
    names = zf.namelist()
    label = next((n for n in names
                  if os.path.basename(n).lower().startswith("ch2_iir")
                  and n.lower().endswith(".xml")
                  and "browse" not in n.lower()), None)
    if label is None:
        return None
    browse = next((n for n in names if n.lower().endswith(".png")), None)
    pid = os.path.basename(label).rsplit(".", 1)[0]
    cache_dir = os.path.join(IIRS_CACHE, pid)
    os.makedirs(cache_dir, exist_ok=True)
    xml_text = zf.read(label).decode("utf-8", "replace")
    with open(os.path.join(cache_dir, "label.xml"), "w",
              encoding="utf-8") as f:
        f.write(xml_text)
    if browse:
        with open(os.path.join(cache_dir, "browse.png"), "wb") as f:
            f.write(zf.read(browse))
    p = _parse_label_text(xml_text, None)
    geo = p["geometry"]
    # ENVI .hdr carries the real cube geometry (the PDS4 label for .qub
    # rasters often omits the Array axis elements)
    if not (p["bands"] and p["lines"] and p["samples"]):
        hdr = next((n for n in names if n.lower().endswith(".hdr")), None)
        if hdr:
            try:
                h = zf.read(hdr).decode("utf-8", "replace")
                def hv(key):
                    m = re.search(r"%s\s*=\s*(\d+)" % key, h, re.I)
                    return int(m.group(1)) if m else 0
                p["bands"] = p["bands"] or hv("bands")
                p["lines"] = p["lines"] or hv("lines")
                p["samples"] = p["samples"] or hv("samples")
            except Exception:                                # noqa: BLE001
                pass
    kind = ("spectral cube (%d bands)" % p["bands"]
            if p.get("bands") else "spectral cube")
    return {
        "product_id": pid,
        "mission": geo.get("mission", "Chandrayaan-2"),
        "instrument": geo.get(
            "instrument", "IIRS (Imaging Infra-Red Spectrometer)"),
        "product_kind": kind,
        "start_time": p["start_time"],
        "browse_png": os.path.join(cache_dir, "browse.png")
        if browse else None,
        "img_path": None, "label_path": None,
        "bands": p["bands"], "lines": p["lines"], "samples": p["samples"],
        "dtype": p["dtype"], "offset": p["offset"],
        "geometry": geo,
        "source": src_note,
        "cached": True,
    }


def _parse_label_text(xml, xml_path):
    """Axis/dtype/offset parse straight off label text (shared by the
    dir-scan and zip-scan paths)."""
    def ax(name):
        m = re.search(r"<axis_name>%s</axis_name>\s*<elements>(\d+)" % name,
                      xml)
        return int(m.group(1)) if m else 0

    dt = re.search(r"<data_type>([^<]+)", xml)
    offs = re.findall(r'<offset unit="byte">(\d+)</offset>', xml)
    p = {
        "lines": ax("Line"), "samples": ax("Sample"), "bands": ax("Band"),
        "dtype": _DTYPE.get(dt.group(1).strip()) if dt else None,
        "offset": int(offs[-1]) if offs else 0,
        "geometry": parse_isda_geometry(xml),
    }
    m = re.search(r"<start_date_time>([^<]+)", xml)
    p["start_time"] = m.group(1).strip() if m else None
    if xml_path:
        img = xml_path.rsplit(".", 1)[0] + ".img"
        p["img_path"] = img if os.path.exists(img) else None
    else:
        p["img_path"] = None
    return p


def _zip_signature():
    """(path, size, mtime) for every IIRS product zip on disk - the library
    rebuilds automatically when any of them changes (e.g. a download that
    just completed)."""
    import glob
    out = []
    for gpat in _PRODUCT_ZIP_GLOBS:
        for zp in glob.glob(gpat):
            try:
                st = os.stat(zp)
                out.append([zp, st.st_size, int(st.st_mtime)])
            except OSError:
                continue
    return out


def _build_library():
    import glob
    os.makedirs(IIRS_CACHE, exist_ok=True)
    entries = {}
    # 1) zip-indexed products - skip re-opening big zips when the cached
    #    label is intact and the zip's (size, mtime) signature is unchanged
    sig = {z[0]: (z[1], z[2]) for z in _zip_signature()}
    old_lib = {}
    if os.path.exists(LIB_JSON):
        try:
            old_lib = json.load(open(LIB_JSON)).get("products", {})
        except Exception:                                    # noqa: BLE001
            old_lib = {}
    for gpat in _PRODUCT_ZIP_GLOBS:
        for zp in sorted(glob.glob(gpat)):
            pid_guess = os.path.basename(zp).rsplit(".", 1)[0]
            cached_label = os.path.join(IIRS_CACHE, pid_guess, "label.xml")
            old = old_lib.get(pid_guess)
            if os.path.exists(cached_label) and old is not None \
                    and old.get("_zip_sig") == sig.get(zp):
                entries[pid_guess] = old
                continue
            try:
                with zipfile.ZipFile(zp) as zf:  # raises mid-download
                    e = _extract_iirs_zip(zf, zp)
                    if e:
                        e["_zip_sig"] = sig.get(zp)
                        entries[e["product_id"]] = e
            except Exception as exc:                     # noqa: BLE001
                print("iirs: skipping %s (download in progress?): %s"
                      % (os.path.basename(zp), str(exc)[:90]))
    # 2) extracted product directories on disk (override zip entries)
    for root in _find_product_dirs():
        xn = next((f for f in os.listdir(root)
                   if f.lower().startswith("ch2_iir")
                   and f.lower().endswith(".xml")), None)
        if not xn:
            continue
        try:
            xml_path = os.path.join(root, xn)
            xml = open(xml_path, encoding="utf-8", errors="replace").read()
            p = _parse_label_text(xml, xml_path)
            pid = xn.rsplit(".", 1)[0]
            browse = None
            for cand_root, _, files in os.walk(os.path.dirname(
                    os.path.dirname(xml_path))):
                for f in files:
                    if f.lower().endswith(".png"):
                        browse = os.path.join(cand_root, f)
                        break
                if browse:
                    break
            entries[pid] = {
                "product_id": pid,
                "mission": (p["geometry"] or {}).get("mission",
                                                     "Chandrayaan-2"),
                "instrument": (p["geometry"] or {}).get(
                    "instrument", "IIRS (Imaging Infra-Red Spectrometer)"),
                "product_kind": "spectral cube (%d bands)" % p["bands"]
                                if p["bands"] else "spectral cube",
                "start_time": p["start_time"],
                "browse_png": browse,
                "img_path": p.get("img_path"), "label_path": xml_path,
                "bands": p["bands"], "lines": p["lines"],
                "samples": p["samples"], "dtype": p["dtype"],
                "offset": p["offset"],
                "geometry": p["geometry"],
                "source": root,
            }
        except Exception:                                    # noqa: BLE001
            continue
    with open(LIB_JSON, "w") as f:
        json.dump({"products": entries, "zip_signature": _zip_signature()},
                  f, indent=1)
    return entries


def _zip_changed():
    """True when any product zip on disk differs from the library snapshot
    (new download appeared / grew / just finished)."""
    if not os.path.exists(LIB_JSON):
        return True
    try:
        old = json.load(open(LIB_JSON)).get("zip_signature", [])
    except Exception:                                        # noqa: BLE001
        return True
    old_map = {e[0]: [e[1], e[2]] for e in old}
    now = {z[0]: [z[1], z[2]] for z in _zip_signature()}
    return old_map != now


def all_products(force_rebuild=False):
    """Indexed IIRS products; [] until IIRS data is on disk.  Rebuilds
    automatically whenever a product zip appears or changes (e.g. the
    PRADAN download just finished)."""
    if force_rebuild or _zip_changed():
        return list(_build_library().values())
    try:
        return list(json.load(open(LIB_JSON)).get("products", {}).values())
    except Exception:                                        # noqa: BLE001
        return list(_build_library().values())


def _product_gray(prod):
    """Grayscale projection of an IIRS product: browse PNG when present,
    else the middle band of the memmapped cube (None when impossible)."""
    if prod.get("browse_png") and os.path.exists(prod["browse_png"]):
        im = cv2.imread(prod["browse_png"], cv2.IMREAD_GRAYSCALE)
        if im is not None:
            return im
    if not (prod.get("img_path") and prod.get("dtype")
            and prod.get("lines") and prod.get("samples")):
        return None
    band = max(0, (prod["bands"] or 1) // 2)
    mm = np.memmap(prod["img_path"], dtype=prod["dtype"], mode="r",
                   offset=prod["offset"],
                   shape=(prod["bands"] or 1, prod["lines"], prod["samples"]))
    sub = np.asarray(mm[band], dtype=np.float32)
    lo, hi = np.percentile(sub, 2), np.percentile(sub, 98)
    return np.clip((sub - lo) / (hi - lo + 1e-9) * 255,
                   0, 255).astype(np.uint8)


def select_best(src_u8, prods, src_center=None):
    """Same ranking contract as tmc.select_best: every product scored by
    footprint proximity + coarse NCC, best-first."""
    small = cv2.resize(src_u8, (256, 256), interpolation=cv2.INTER_AREA)
    cands = []
    for prod in prods:
        ncc, loc, scale = -1.0, None, None
        gray = _product_gray(prod)
        if gray is not None:
            th = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA) \
                if max(gray.shape) > 512 else gray
            best = (-1.0, None, None)
            for s in (0.12, 0.24, 0.48):
                tsz = max(24, int(512 * s))
                if tsz >= th.shape[0] or tsz >= th.shape[1]:
                    continue
                tpl = cv2.resize(small, (tsz, tsz),
                                 interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(th, tpl, cv2.TM_CCOEFF_NORMED)
                _, sc, _, mloc = cv2.minMaxLoc(res)
                if sc > best[0]:
                    best = (float(sc), mloc, s)
            ncc, loc, scale = best
        km = footprint_distance(prod, src_center)
        geo_score = (1.0 / (1.0 + km / 50.0)) if km is not None else 0.0
        cands.append((prod, {"loc": loc, "scale": scale,
                             "coarse_ncc": round(ncc, 3),
                             "footprint_km": (round(km, 1)
                                              if km is not None else None)},
                      0.6 * geo_score + 0.4 * max(0.0, ncc)))
    cands.sort(key=lambda c: -c[2])
    return cands


def build_reference(src_u8, prod, cand, out=1024):
    """Registered 1024^2 IIRS reference: grayscale projection cropped around
    the coarse match, resampled + template-aligned onto the source grid.
    SIFT is skipped for hyperspectral projections (cross-modal - the MI
    statistic is the honest similarity measure here)."""
    gray = _product_gray(prod)
    if gray is None:
        return None, {"error": "no readable IIRS raster/browse for %s"
                              % prod["product_id"]}
    loc, scale = cand.get("loc"), cand.get("scale")
    if loc is None:
        loc = (gray.shape[1] // 4, gray.shape[0] // 4)
        scale = 0.24
    tsz = max(24, int(512 * scale))
    half = max(tsz, 32) * 2
    cx = int(min(max(loc[0], half), max(gray.shape[1] - half - 1, half)))
    cy = int(min(max(loc[1], half), max(gray.shape[0] - half - 1, half)))
    crop = gray[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half]
    region = cv2.resize(crop, (out, out), interpolation=cv2.INTER_CUBIC)
    t = 512
    tpl = region[(out - t) // 2:(out + t) // 2, (out - t) // 2:(out + t) // 2]
    res = cv2.matchTemplate(src_u8, tpl, cv2.TM_CCOEFF_NORMED)
    _, sc, _, loc2 = cv2.minMaxLoc(res)
    dx, dy = loc2[0] - (out - t) // 2, loc2[1] - (out - t) // 2
    region = cv2.warpAffine(region, np.float32([[1, 0, -dx], [0, 1, -dy]]),
                            (out, out))
    a = region.astype(np.float32)
    b = cv2.resize(src_u8, (out, out)).astype(np.float32)
    try:
        import pipeline
        mi = round(float(pipeline.mutual_information(a, b, bins=32)), 3)
    except Exception:                                        # noqa: BLE001
        mi = None
    meta = {
        "product_id": prod["product_id"],
        "mission": prod.get("mission"),
        "instrument": prod.get("instrument"),
        "product_kind": prod.get("product_kind"),
        "footprint_km": cand.get("footprint_km"),
        "coarse_ncc": cand.get("coarse_ncc"),
        "translation": {"dx": int(dx), "dy": int(dy),
                        "ncc": round(float(sc), 3)},
        "mutual_information": mi,
        "cross_modal_note": "IIRS spectral projection vs panchromatic "
                            "OHRC radiance - mutual information is the "
                            "governing cross-modal statistic (Problem 6)",
        "sun": (prod.get("geometry") or {}).get("sun"),
        "scale_note": "grayscale projection resampled onto the OHRC "
                      "1024^2 analysis grid (template-aligned)",
    }
    return region, meta
