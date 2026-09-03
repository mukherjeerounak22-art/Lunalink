"""LROC NAC CDR ingest (Route B, real NASA PDS4 products).

Each product = PDS4 XML label + IMG (5064-byte PDS3 header + int16 array,
Scaled I/F, missing = -32768). Labels carry no footprint coordinates, so
overlap with the OHRC footprint is determined empirically: thumbnail each
product, SIFT-match against the OHRC source, keep the best-corresponding
product, then build a registered 1024^2 reference from it.
"""
import json
import os
import re

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LRO_DIR = os.path.join(ROOT, "data", "reference", "lro_nac")


def parse_product(xml_path):
    xml = open(xml_path, encoding="utf-8").read()

    def g(tag, default=None):
        m = re.search(r"<%s>([^<]+)" % tag, xml)
        return m.group(1).strip() if m else default

    lines = int(g("lines", 0))
    samples = int(g("samples", 0))
    if not lines:  # Axis_Array fallback
        m = re.search(r"<axis_name>Line</axis_name>\s*<elements>(\d+)", xml)
        lines = int(m.group(1)) if m else 0
        m = re.search(r"<axis_name>Sample</axis_name>\s*<elements>(\d+)", xml)
        samples = int(m.group(1)) if m else 0
    img = xml_path.replace(".xml", ".IMG")
    return {
        "product_id": os.path.basename(xml_path).replace(".xml", ""),
        "img_path": img,
        "lines": lines,
        "samples": samples,
        "offset": 5064,                      # PDS3 attached header
        "scaling": 3.05185094759972e-05,     # DN -> I/F
        "missing": -32768,
        "start": g("start_date_time"),
        "stop": g("stop_date_time"),
    }


def all_products():
    if not os.path.isdir(LRO_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(LRO_DIR)):
        if fn.lower().endswith(".xml"):
            try:
                p = parse_product(os.path.join(LRO_DIR, fn))
                if p["lines"] and os.path.exists(p["img_path"]):
                    out.append(p)
            except Exception:
                continue
    return out


def ensure_library():
    """One-time fetch of the LRO NAC strip library from a PUBLIC Kaggle
    dataset when the local library is empty - this is what keeps free-tier
    deploys (Render free: 512 MB RAM, ephemeral disk) capable of REAL NASA
    auto-selection for uploaded scenes.  Set
        KAGGLE_LRO_DATASET=<owner>/<dataset-slug>
    (dataset contains the *.IMG + *.xml pairs, at its root or any
    subfolder; no Kaggle credentials needed for public datasets).  The
    strips are np.memmap-read, so RAM stays flat; files are symlinked out
    of the kagglehub cache so only one copy exists on disk.  Any failure
    returns False and the caller degrades to a simulated second pass."""
    if all_products():
        return True
    slug = os.environ.get("KAGGLE_LRO_DATASET", "").strip()
    if not slug:
        return False
    try:
        import kagglehub
        print("lroc: strip library empty - pulling %s from Kaggle "
              "(one-time per instance, ~3 GB; honest cold-start cost)"
              % slug)
        base = kagglehub.dataset_download(slug)
        os.makedirs(LRO_DIR, exist_ok=True)
        n = 0
        for root, _, files in os.walk(base):
            for fn in files:
                if not fn.lower().endswith((".img", ".xml")):
                    continue
                src = os.path.join(root, fn)
                dst = os.path.join(LRO_DIR, fn)
                if os.path.exists(dst):
                    n += 1
                    continue
                try:
                    os.symlink(src, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
                n += 1
        print("lroc: linked %d strip files from the Kaggle cache" % n)
        return bool(all_products())
    except Exception as exc:                                 # noqa: BLE001
        print("lroc: Kaggle fetch failed (%s) - uploads will fall back "
              "to a simulated second pass" % str(exc)[:140])
        return False


def coarse_preview(prod, factor=8):
    """Coarse preview of the whole NAC strip at ~factor NAC pixels per cell
    (factor=8 -> ~4 m/cell). Returns uint8 (lines/factor, samples/factor)."""
    mm = np.memmap(prod["img_path"], dtype="<i2", mode="r",
                   offset=prod["offset"],
                   shape=(prod["lines"], prod["samples"]))
    sub = np.asarray(mm[::factor, ::factor], dtype=np.float32)
    sub[sub == prod["missing"]] = np.nan
    lo, hi = np.nanpercentile(sub, 2), np.nanpercentile(sub, 98)
    v8 = np.clip((sub - lo) / (hi - lo + 1e-9) * 255, 0, 255)
    return np.nan_to_num(v8).astype(np.uint8)


def select_best(src_u8, prods, factor=8):
    """Locate the OHRC patch inside each NAC strip; returns ALL candidates
    [(prod, loc, score)] sorted by coarse NCC (desc). Final selection is
    done by post-alignment translation NCC in the caller."""
    candidates = []
    for prod in prods:
        try:
            prev = coarse_preview(prod, factor)
            # OHRC patch footprint at the preview's meters-per-cell scale
            cell_m = 0.5 * factor / 2.0  # NAC ~0.5 m/px, 2x2 preview cell
            size = int(round(1024 * 4 * 0.26488811295333897 / cell_m))
            size = min(size, prev.shape[0], prev.shape[1])
            templ = cv2.resize(src_u8, (size, size),
                               interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(prev, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            print("  LROC %s: preview %s template %d px, NCC %.3f at %s"
                  % (prod["product_id"], prev.shape, size, score, loc))
            candidates.append((prod, loc, score))
        except Exception as exc:
            print("  LROC %s skipped: %s" % (prod["product_id"], exc))
    candidates.sort(key=lambda c: -c[2])
    return candidates


def build_reference(src_u8, prod, loc, factor=8, out=1024):
    """Extract the full-resolution NAC region around the matched location,
    resample it onto the OHRC 1024^2 grid (~1.06 m/px), then SIFT-refine
    the alignment. Returns (ref_u8, meta) or (None, error-meta)."""
    # full-res footprint of the OHRC patch in NAC pixels
    side_full = int(round(1024 * 4 * 0.26488811295333897 / 0.5))  # ~2170
    cx_full = int((loc[0] + 0.5) * factor)      # preview cell center
    cy_full = int((loc[1] + 0.5) * factor)
    half = side_full // 2 + 60
    r0 = max(cy_full - half, 0)
    c0 = max(cx_full - half, 0)
    r1 = min(cy_full + half, prod["lines"])
    c1 = min(cx_full + half, prod["samples"])
    if r1 - r0 < 256 or c1 - c0 < 256:
        return None, {"error": "overlap region too small"}

    mm = np.memmap(prod["img_path"], dtype="<i2", mode="r",
                   offset=prod["offset"],
                   shape=(prod["lines"], prod["samples"]))
    sub = np.asarray(mm[r0:r1, c0:c1], dtype=np.float32)
    sub[sub == prod["missing"]] = np.nan
    lo, hi = np.nanpercentile(sub, 2), np.nanpercentile(sub, 98)
    sub8 = np.clip((sub - lo) / (hi - lo + 1e-9) * 255, 0, 255)
    sub8 = np.nan_to_num(sub8).astype(np.uint8)

    # resample region onto the OHRC grid scale (~1.06 m/px): NAC region
    # spans side_full NAC px over the same ground as 1024 OHRC-grid px
    region = cv2.resize(sub8, (out, out), interpolation=cv2.INTER_AREA)

    # translation pre-alignment: locate the region's central block inside
    # the OHRC source, then shift so both frames coincide
    trans = None
    t = 512
    tpl = region[(out - t) // 2:(out + t) // 2, (out - t) // 2:(out + t) // 2]
    res = cv2.matchTemplate(src_u8, tpl, cv2.TM_CCOEFF_NORMED)
    _, sc, _, loc2 = cv2.minMaxLoc(res)
    dx, dy = loc2[0] - (out - t) // 2, loc2[1] - (out - t) // 2
    region = cv2.warpAffine(region, np.float32([[1, 0, -dx], [0, 1, -dy]]),
                            (out, out))
    trans = {"dx": int(dx), "dy": int(dy), "ncc": round(float(sc), 3)}

    # SIFT refinement: align region to the OHRC source grid
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    k1, d1 = sift.detectAndCompute(src_u8, None)
    k2, d2 = sift.detectAndCompute(region, None)
    H = None
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
    inliers = int(mask.sum()) if H is not None else 0
    if H is not None:
        region = cv2.warpPerspective(region, H, (out, out),
                                     borderValue=0)
    meta = {"product_id": prod["product_id"], "inliers": inliers,
            "sift_refined": H is not None,
            "translation": trans,
            "region_full_px": [c0, r0, c1, r1],
            "start": prod["start"], "stop": prod["stop"],
            "scale_note": "NAC CDR (~0.5 m/px) resampled onto the OHRC "
                          "1024^2 analysis grid (~1.06 m/px)"
                          + (f" and SIFT-refined ({inliers} inliers)"
                             if H is not None else " (template-aligned)")}
    return region, meta

