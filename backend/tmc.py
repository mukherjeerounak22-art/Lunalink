"""TMC / TMC-2 DTM product library - the ISRO cross-instrument reference route.

The PRADAN TMC bundles are plain TARs of per-product ZIPs; each product ZIP
carries a huge GeoTIFF DTM plus small metadata: the PDS4 XML label (ISDA
footprint corners + sun angles) and a browse PNG (shaded-relief preview).
This library indexes every product WITHOUT extracting the DTM rasters:
label + browse PNG are lazily cache-extracted to data/reference/tmc/<pid>/
so nearest-reference auto-selection runs on any host (including a small
deployed instance) at zero raster cost.

Selection mirrors backend/lroc.py (the NASA route):
  rank every product by great-circle footprint distance + coarse NCC
  template match on the browse thumbnail, then build a registered 1024^2
  reference (template pre-alignment + SIFT refinement).  Cross-modal note:
  TMC DTM shaded relief vs OHRC panchromatic radiance - mutual information
  (pipeline.mutual_information, the Problem-6 statistic) is reported
  alongside NCC in the returned metadata.
"""
import io
import json
import math
import os
import re
import tarfile
import zipfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMC_DIR = os.path.join(ROOT, "data", "reference", "tmc")
LIB_JSON = os.path.join(TMC_DIR, "_library.json")

# bundle tars / product zips are looked for in exactly these places
_BUNDLE_GLOBS = [
    os.path.join(ROOT, "ch[12]_tmc_ndn_*Bundle*.tar"),
    os.path.join(ROOT, "data", "raw", "tmc_bundles", "*.tar"),
    os.path.join(ROOT, "data", "raw", "tmc2", "*.tar"),
]
_PRODUCT_ZIP_GLOBS = [
    os.path.join(ROOT, "ch[12]_tmc_ndn_*.zip"),
    os.path.join(ROOT, "data", "raw", "tmc_bundles", "*.zip"),
    os.path.join(ROOT, "data", "raw", "tmc2", "*.zip"),
]
# already-extracted product directories (label + tif/img + optional browse)
_EXTRACTED_DIRS = [
    os.path.join(ROOT, "data", "raw", "tmc2"),
    os.path.join(ROOT, "data", "UNZIPPED_DATA"),
]


def parse_isda_geometry(xml_text):
    """Footprint corners + sun geometry out of ANY ISRO PDS4 label (the
    isda namespace block).  Longitudes are 0-360 East as PRADAN ships them.
    Returns {} for labels without the block (e.g. NASA PDS3)."""
    def g(pat):
        m = re.search(pat, xml_text)
        return float(m.group(1)) if m else None

    corners = {}
    for cn in ("upper_left", "upper_right", "lower_left", "lower_right"):
        lat = g(r"<isda:%s_latitude[^>]*>([-\d.eE+]+)" % cn)
        lon = g(r"<isda:%s_longitude[^>]*>([-\d.eE+]+)" % cn)
        if lat is not None and lon is not None:
            corners[cn] = {"lat_deg": lat, "lon_deg": lon}
    out = {}
    if corners:
        out["footprint_corners"] = corners
        lats = [c["lat_deg"] for c in corners.values()]
        lons = [c["lon_deg"] for c in corners.values()]
        out["center"] = {"lat_deg": sum(lats) / len(lats),
                         "lon_deg": sum(lons) / len(lons)}
    az, el = g(r"<isda:sun_azimuth[^>]*>([-\d.eE+]+)"), \
        g(r"<isda:sun_elevation[^>]*>([-\d.eE+]+)")
    if az is not None and el is not None:
        out["sun"] = {"sun_azimuth_deg": az, "sun_elevation_deg": el}
    m = re.search(r"<start_date_time>([^<]+)", xml_text)
    if m:
        out["start_time"] = m.group(1).strip()
    # mission + instrument out of the observing-system block
    m = re.search(r"<Investigation_Area>.*?<name>([^<]+)", xml_text, re.S)
    if m:
        out["mission"] = m.group(1).strip()
    for blk in re.findall(r"<Observing_System_Component>.*?</"
                          "Observing_System_Component>", xml_text, re.S):
        if "<type>Instrument</type>" in blk:
            mi = re.search(r"<name>([^<]+)", blk)
            if mi:
                out["instrument"] = mi.group(1).strip()
            break
    return out


def great_circle_km(lat1, lon1, lat2, lon2):
    """Haversine great-circle distance in km (spherical moon, R=1737.4 km)."""
    R = 1737.4
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians((lon2 - lon1) % 360.0)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * \
        math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _open_inner_zip(stream):
    """ZipFile over a tar member (seekable for plain tars); full-read
    fallback for exotic archives."""
    try:
        return zipfile.ZipFile(stream)
    except Exception:                                        # noqa: BLE001
        return zipfile.ZipFile(io.BytesIO(stream.read()))


def _extract_from_product_zip(zf, base_dir, src_note):
    """Pull label + browse PNG out of one product ZIP; parse the ISDA
    geometry; return the library entry (or None for non-products)."""
    names = zf.namelist()
    label = next((n for n in names
                  if n.lower().endswith(".xml")
                  and "data/" in n.replace("\\", "/").lower()), None)
    if label is None:   # browse-only or non-product zip
        return None
    browse = next((n for n in names if n.lower().endswith(".png")
                   and "browse" in n.lower()), None)
    pid = os.path.basename(label).rsplit(".", 1)[0]
    cache_dir = os.path.join(base_dir, pid)
    os.makedirs(cache_dir, exist_ok=True)
    xml_text = zf.read(label).decode("utf-8", "replace")
    with open(os.path.join(cache_dir, "label.xml"), "w",
              encoding="utf-8") as f:
        f.write(xml_text)
    if browse:
        with open(os.path.join(cache_dir, "browse.png"), "wb") as f:
            f.write(zf.read(browse))
    geo = parse_isda_geometry(xml_text)
    return {
        "product_id": pid,
        "mission": geo.get("mission",
                           "Chandrayaan-2" if pid.startswith("ch2")
                           else "Chandrayaan-1"),
        "instrument": geo.get("instrument",
                              "TMC (Terrain Mapping Camera)"),
        "product_kind": "DTM (derived)",
        "start_time": geo.get("start_time"),
        "cache_dir": cache_dir,
        "browse_png": os.path.join(cache_dir, "browse.png")
        if browse else None,
        "geometry": geo,
        "source": src_note,
    }


def _index_extracted_dir(base):
    """Index label+tif (or img) product dirs already unpacked on disk."""
    out = []
    for root, _, files in os.walk(base):
        if "browse" in root.lower() or "miscellaneous" in root.lower():
            continue
        for xn in [f for f in files if f.lower().endswith(".xml")]:
            if not any(f.lower().endswith((".tif", ".tiff", ".img"))
                       for f in files):
                continue
            xp = os.path.join(root, xn)
            pid = xn.rsplit(".", 1)[0]
            cache_dir = os.path.join(TMC_DIR, pid)
            browse = next((os.path.join(root, f) for f in files
                           if f.lower().endswith(".png")), None)
            try:
                geo = parse_isda_geometry(open(xp, encoding="utf-8",
                                               errors="replace").read())
            except Exception:                                # noqa: BLE001
                geo = {}
            out.append({
                "product_id": pid,
                "mission": geo.get("mission",
                                   "Chandrayaan-2" if pid.startswith("ch2")
                                   else "Chandrayaan-1"),
                "instrument": geo.get("instrument",
                                      "TMC (Terrain Mapping Camera)"),
                "product_kind": "DTM (derived)",
                "start_time": geo.get("start_time"),
                "cache_dir": cache_dir,
                "browse_png": browse,
                "geometry": geo,
                "source": root,
            })
    return out


def _build_library():
    """Scan every bundle tar / product zip / extracted dir; cache-extract
    the small files; dedupe by product_id.  Truncated downloads are
    recorded and skipped (the audit showed 3 of the 9 tars are cut off)."""
    import glob
    os.makedirs(TMC_DIR, exist_ok=True)
    entries, warnings = {}, []
    for gpat in _BUNDLE_GLOBS:
        for tarp in sorted(glob.glob(gpat)):
            try:
                with tarfile.open(tarp, "r:*") as t:
                    for m in t:
                        if not m.name.lower().endswith(".zip"):
                            continue
                        try:
                            zf = _open_inner_zip(t.extractfile(m))
                            e = _extract_from_product_zip(
                                zf, TMC_DIR, os.path.basename(tarp))
                        except Exception as exc:             # noqa: BLE001
                            warnings.append("%s/%s: %s"
                                            % (os.path.basename(tarp),
                                               m.name, exc))
                            continue
                        if e:
                            entries.setdefault(e["product_id"], e)
            except Exception as exc:                         # noqa: BLE001
                warnings.append("%s: unreadable (%s)"
                                % (os.path.basename(tarp), exc))
    for gpat in _PRODUCT_ZIP_GLOBS:
        for zp in sorted(glob.glob(gpat)):
            try:
                with zipfile.ZipFile(zp) as zf:
                    e = _extract_from_product_zip(
                        zf, TMC_DIR, os.path.basename(zp))
                    if e:
                        entries.setdefault(e["product_id"], e)
            except Exception as exc:                         # noqa: BLE001
                warnings.append("%s: unreadable (%s)"
                                % (os.path.basename(zp), exc))
    for base in _EXTRACTED_DIRS:
        if os.path.isdir(base):
            for e in _index_extracted_dir(base):
                # only TMC-family products; other instruments (OHRC) are
                # sources, not reference-library members
                if re.match(r"ch[12]_tmc_", e["product_id"]):
                    entries.setdefault(e["product_id"], e)
    for pid, e in entries.items():
        if not e["browse_png"] or not os.path.exists(e["browse_png"]):
            e["browse_png"] = None
    lib = {"products": entries, "warnings": warnings[:40]}
    with open(LIB_JSON, "w") as f:
        json.dump(lib, f, indent=1)
    return lib


def all_products(force_rebuild=False):
    """The full TMC/TMC-2 product library (browse thumbnails + labels only,
    DTMs never extracted).  Cached in data/reference/tmc/_library.json."""
    if force_rebuild or not os.path.exists(LIB_JSON):
        lib = _build_library()
    else:
        try:
            lib = json.load(open(LIB_JSON))
        except Exception:                                    # noqa: BLE001
            lib = _build_library()
    return list(lib.get("products", {}).values())


def library_warnings():
    if os.path.exists(LIB_JSON):
        return json.load(open(LIB_JSON)).get("warnings", [])
    return []


def _browse_gray(prod):
    """Cached browse PNG as uint8 grayscale (or None)."""
    p = prod.get("browse_png")
    if p and os.path.exists(p):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            return im
    return None


def _valid_bbox(gray):
    """Bounding box of REAL content inside a PDS4 browse thumbnail - the
    white letterbox margins and black nodata edges are excluded so a
    reference crop can never land on a blank field."""
    h, w = gray.shape
    m = (gray > 10) & (gray < 245)
    if m.mean() >= 0.97:                       # no margins worth trimming
        return 0, w, 0, h
    rows = np.where(m.sum(axis=1) > w * 0.25)[0]
    cols = np.where(m.sum(axis=0) > h * 0.25)[0]
    if len(rows) < 8 or len(cols) < 8:
        return 0, w, 0, h
    return int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1


def footprint_distance(prod, src_center):
    """Great-circle km between the product footprint center and the source
    center dict {lat_deg, lon_deg} (None when either lacks coordinates)."""
    c = (prod.get("geometry") or {}).get("center")
    if not c or not src_center:
        return None
    return great_circle_km(src_center["lat_deg"], src_center["lon_deg"],
                           c["lat_deg"], c["lon_deg"])


def select_best(src_u8, prods, src_center=None):
    """Rank products for a source patch.  Returns [(prod, cand, score)]
    sorted best-first; the score blends footprint proximity (great-circle
    km, when both footprints are known) with the best multi-scale coarse
    NCC of the downscaled source inside the browse thumbnail.  Every
    product gets a candidate entry so the caller can report the full
    sorted ranking - 'sorted and selected altogether'."""
    small = cv2.resize(src_u8, (256, 256), interpolation=cv2.INTER_AREA)
    cands = []
    for prod in prods:
        ncc, loc, scale = -1.0, None, None
        thumb = _browse_gray(prod)
        if thumb is not None:
            th = cv2.resize(thumb, (512, 512), interpolation=cv2.INTER_AREA) \
                if max(thumb.shape) > 512 else thumb
            best = (-1.0, None, None)
            for s in (0.12, 0.24, 0.48):   # template sizes as thumb fraction
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
            if loc is not None:
                # rescale the match location from the (possibly 512-
                # resized) working image back to ORIGINAL thumb pixels -
                # build_reference crops from the original, and an
                # un-rescaled location lands on the letterbox margins
                # (the blank-white-reference bug)
                sy_ = thumb.shape[0] / float(th.shape[0])
                sx_ = thumb.shape[1] / float(th.shape[1])
                loc = (int(loc[0] * sx_), int(loc[1] * sy_))
        km = footprint_distance(prod, src_center)
        geo_score = (1.0 / (1.0 + km / 50.0)) if km is not None else 0.0
        score = 0.6 * geo_score + 0.4 * max(0.0, ncc)
        cands.append((prod, {"loc": loc, "scale": scale,
                             "thumb_shape": [int(thumb.shape[0]),
                                             int(thumb.shape[1])]
                             if thumb is not None else None,
                             "coarse_ncc": round(ncc, 3),
                             "footprint_km": (round(km, 1)
                                              if km is not None else None)},
                      score))
    cands.sort(key=lambda c: -c[2])
    return cands


def build_reference(src_u8, prod, cand, out=1024):
    """Registered 1024^2 TMC reference for the source patch: crop the browse
    thumbnail around the coarse match, resample onto the source grid, then
    template + SIFT refinement (cross-modal: SIFT often fails on DTM-shaded
    relief vs panchromatic radiance - template alignment is the fallback,
    reported honestly).  Returns (region_u8, meta)."""
    thumb = _browse_gray(prod)
    if thumb is None:
        return None, {"error": "no browse thumbnail cached for %s"
                              % prod["product_id"]}
    loc, scale = cand.get("loc"), cand.get("scale")
    th_hw = cand.get("thumb_shape") or list(thumb.shape)
    if loc is None:                     # no coarse location - valid center
        loc = (thumb.shape[1] // 2, thumb.shape[0] // 2)
        scale = 0.24
    # template size was a fraction of the 512 working image -> convert to
    # ORIGINAL thumb pixels (th_hw records the working-image size used)
    tsz = max(24, int(512 * scale) * thumb.shape[1] // max(int(th_hw[1]), 1))
    half = max(tsz, 32) * 2
    # never crop into the white letterbox / black nodata margins
    x0, x1, y0, y1 = _valid_bbox(thumb)

    def _crop(cx, cy):
        return thumb[max(cy - half, 0):cy + half,
                     max(cx - half, 0):cx + half]

    def _valid_frac(cx, cy):
        c = _crop(cx, cy)
        if c.size == 0:
            return 0.0
        return float(((c > 10) & (c < 245)).mean())

    cx = int(np.clip(loc[0], x0, x1 - 1))
    cy = int(np.clip(loc[1], y0, y1 - 1))
    if _valid_frac(cx, cy) < 0.6:
        # the coarse (cross-modal) match landed on a margin - search the
        # valid box for the densest real-content window instead
        best_s, cx, cy = -1.0, cx, cy
        for fy in np.linspace(0.1, 0.9, 6):
            for fx in np.linspace(0.1, 0.9, 6):
                gx = int(x0 + (x1 - x0) * fx)
                gy = int(y0 + (y1 - y0) * fy)
                s = _valid_frac(gx, gy)
                if s > best_s:
                    best_s, cx, cy = s, gx, gy
    crop = _crop(cx, cy)
    region = cv2.resize(crop, (out, out), interpolation=cv2.INTER_CUBIC)

    # template pre-alignment onto the source grid (same scheme as lroc.py)
    t = 512
    tpl = region[(out - t) // 2:(out + t) // 2, (out - t) // 2:(out + t) // 2]
    res = cv2.matchTemplate(src_u8, tpl, cv2.TM_CCOEFF_NORMED)
    _, sc, _, loc2 = cv2.minMaxLoc(res)
    dx, dy = loc2[0] - (out - t) // 2, loc2[1] - (out - t) // 2
    region = cv2.warpAffine(region, np.float32([[1, 0, -dx], [0, 1, -dy]]),
                            (out, out))
    trans = {"dx": int(dx), "dy": int(dy), "ncc": round(float(sc), 3)}

    # SIFT refinement (cross-modal best effort)
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
    k1, d1 = sift.detectAndCompute(src_u8, None)
    k2, d2 = sift.detectAndCompute(region, None)
    H, inliers = None, 0
    if d1 is not None and d2 is not None:
        raw = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        good = [pr[0] for pr in raw
                if len(pr) == 2 and pr[0].distance < 0.8 * pr[1].distance]
        if len(good) >= 10:
            pts1 = np.float32([k1[m.queryIdx].pt for m in good])
            pts2 = np.float32([k2[m.trainIdx].pt for m in good])
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 6.0)
            if H is None or int(mask.sum()) < 10:
                H = None
            else:
                inliers = int(mask.sum())
    if H is not None:
        region = cv2.warpPerspective(region, H, (out, out), borderValue=0)

    # FINAL contrast normalisation (after all warps, so warp borders and
    # save_png cannot re-blow the histogram): gain-capped, mean-centred -
    # a full percentile stretch turns plateau-heavy DTM histograms into a
    # washed-out white field
    lo, hi = np.percentile(region, 2), np.percentile(region, 98)
    if hi - lo > 1e-6:
        gain = min(255.0 / (hi - lo), 1.7)
        r = (region.astype(np.float32) - lo) * gain
        r += (128.0 - r.mean())
        region = np.clip(r, 0, 255).astype(np.uint8)

    # cross-modal similarity: NCC + mutual information (Problem-6 statistic)
    a = region.astype(np.float32)
    b = cv2.resize(src_u8, (out, out)).astype(np.float32)
    ncc_reg = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) \
        if np.std(a) > 1e-6 and np.std(b) > 1e-6 else 0.0
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
        "footprint_center": (prod.get("geometry") or {}).get("center"),
        "coarse_ncc": cand.get("coarse_ncc"),
        "translation": trans,
        "sift_refined": H is not None,
        "inliers": inliers,
        "post_alignment_ncc": round(ncc_reg, 3),
        "mutual_information": mi,
        "cross_modal_note": "TMC DTM shaded-relief browse vs panchromatic "
                            "OHRC radiance - different sensor physics; "
                            "mutual information reported alongside NCC",
        "sun": (prod.get("geometry") or {}).get("sun"),
        "scale_note": "browse thumbnail resampled onto the OHRC 1024^2 "
                      "analysis grid" + (f" and SIFT-refined ({inliers} "
                                         "inliers)" if H is not None
                                         else " (template-aligned)"),
    }
    return region, meta


def ranked_summaries(src_u8, src_center=None, top=5):
    """The 'sorted' list the UI shows: every library product ranked by
    footprint proximity + coarse NCC, best-first."""
    prods = all_products()
    if not prods:
        return [], "TMC library empty - no TMC bundle found on disk"
    cands = select_best(src_u8, prods, src_center)
    out = []
    for prod, cand, score in cands[:top]:
        out.append({
            "product_id": prod["product_id"],
            "mission": prod.get("mission"),
            "instrument": prod.get("instrument"),
            "footprint_km": cand.get("footprint_km"),
            "coarse_ncc": cand.get("coarse_ncc"),
            "score": round(float(score), 3),
        })
    return out, None
