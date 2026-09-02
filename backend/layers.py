"""Multi-instrument data layers - the fusion step of the three-instrument
stack (OHRC backbone / TMC-2 metric stereo / IIRS mineralogy).

Everything here is lazily computed and disk-cached per product:
  - IIRS mineral classification: the .qub spectral cube (BSQ float32) is
    stream-extracted ONCE from the product zip, memmapped, stride-sampled,
    continuum-removed over the diagnostic regions and classified by band
    depth heuristics (1 um crystal-field band, 2 um band, 3 um OH/H2O).
    Honest note: rule-based band-depth classes, NOT laboratory-spectrum
    matching (SAM against RELAB/M3 is future work).
  - TMC-2 metric DEM: the stereo-photogrammetry GeoTIFF is extracted ONCE,
    read as float32 metric heights (m) and downsampled onto the 192^2
    terrain grid - true measured heights, which also enable the
    SFS-vs-metric validation difference map.
No raster is ever kept in RAM beyond one downsample pass.
"""
import json
import os
import re
import tarfile
import zipfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IIRS_CACHE = os.path.join(ROOT, "data", "reference", "iirs")
TMC_CACHE = os.path.join(ROOT, "data", "reference", "tmc")
PROC = os.path.join(ROOT, "data", "processed")

GRID_N = 192   # must match main.GRID_N

# heuristic rule-based mineral classes (legend shared with the frontend)
MINERAL_LEGEND = [
    {"id": 0, "name": "pyroxene-rich (mafic)", "color": "#d64545",
     "rule": "deep 1 um + deep 2 um crystal-field bands"},
    {"id": 1, "name": "olivine-rich (mafic)", "color": "#3fa66a",
     "rule": "broad 1 um band, weak 2 um band"},
    {"id": 2, "name": "feldspathic (highland-like)", "color": "#cccccc",
     "rule": "shallow 1 um and 2 um bands (low Fe bearing phases)"},
    {"id": 3, "name": "mixed / immature regolith", "color": "#c8a24a",
     "rule": "intermediate band depths"},
    {"id": 4, "name": "3 um OH/H2O-bearing", "color": "#4a7fc8",
     "rule": "distinct absorption drop near 3 um"},
    {"id": 5, "name": "low signal / uncertain", "color": "#333333",
     "rule": "reflectance too low or saturated for band-depth analysis"},
]


def _extract_member_once(zip_path, member_suffix, out_path):
    """Stream one member (by suffix) out of a product zip exactly once."""
    if os.path.exists(out_path):
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist()
                    if n.lower().endswith(member_suffix.lower()))
        with zf.open(name) as src, open(tmp, "wb") as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    os.replace(tmp, out_path)
    return out_path


def _wavelength(band_index):
    """Approximate IIRS channel center (um): 256 channels spanning
    0.8-5.0 um, near-uniform ~16.4 nm/channel (heuristic)."""
    return 0.8 + band_index * (4.2 / 255.0)


def _band_for(wavelength_um):
    return int(round((wavelength_um - 0.8) / (4.2 / 255.0)))


def _continuum_removed_depth(refl, lo_um, hi_um):
    """Band depth after linear continuum removal over [lo_um, hi_um].
    refl: (bands,) reflectance of one spectrum. Returns (depth, center
    wavelength) using the shoulder-to-shoulder line as the continuum."""
    i0, i1 = _band_for(lo_um), _band_for(hi_um)
    if i1 - i0 < 3 or i1 >= len(refl):
        return 0.0, 0.0
    seg = refl[i0:i1 + 1].astype(np.float64)
    if not np.isfinite(seg).all() or seg.max() <= 0:
        return 0.0, 0.0
    base = np.linspace(seg[0], seg[-1], seg.size)
    base = np.clip(base, 1e-6, None)
    ratio = seg / base
    k = int(np.argmin(ratio))
    depth = float(1.0 - ratio[k])
    return (depth if np.isfinite(depth) else 0.0), \
        float(_wavelength(i0 + k))


def iirs_minerals(product, stride=64, force=False):
    """Classified mineral map for an IIRS product (classes HxW uint8 +
    meta), cached to data/reference/iirs/<pid>/minerals.npz."""
    pid = product["product_id"]
    cache_npz = os.path.join(IIRS_CACHE, pid, "minerals.npz")
    if os.path.exists(cache_npz) and not force:
        d = np.load(cache_npz, allow_pickle=True)
        return d["classes"], dict(d["meta"].item())
    zip_path = product.get("source") \
        if str(product.get("source", "")).lower().endswith(".zip") else None
    cube = os.path.join(IIRS_CACHE, pid, "cube.qub")
    if zip_path and not os.path.exists(zip_path):
        # relative/basename source - resolve against the known locations
        import glob
        for pat in (os.path.join(ROOT, "*.zip"),
                    os.path.join(ROOT, "data", "raw", "iirs", "*.zip")):
            m = [p for p in glob.glob(pat)
                 if os.path.basename(p) == os.path.basename(zip_path)]
            if m:
                zip_path = m[0]
                break
    if zip_path and not os.path.exists(cube):
        cube = _extract_member_once(zip_path, ".qub", cube)
    if not os.path.exists(cube):
        return None, {"error": "spectral cube not available for %s" % pid}
    bands = product.get("bands") or 256
    lines = product.get("lines") or 0
    samples = product.get("samples") or 0
    if not (bands and lines and samples):
        hdr = os.path.join(IIRS_CACHE, pid, "geometry.hdr")
        if os.path.exists(hdr):
            h = open(hdr, encoding="utf-8", errors="replace").read()
            for key, slot in (("bands", "b"), ("lines", "l"),
                              ("samples", "s")):
                m = re.search(r"%s\s*=\s*(\d+)" % key, h, re.I)
                if m:
                    v = int(m.group(1))
                    if slot == "b":
                        bands = bands or v
                    elif slot == "l":
                        lines = lines or v
                    else:
                        samples = samples or v
    if not (bands and lines and samples):
        return None, {"error": "cube geometry unknown for %s" % pid}
    mm = np.memmap(cube, dtype="<f4", mode="r",
                   shape=(bands, lines, samples))
    rows = list(range(0, lines, stride))
    cols = list(range(0, samples, max(1, samples // 128)))
    classes = np.full((len(rows), len(cols)), 5, dtype=np.uint8)
    feats = np.zeros((len(rows), len(cols), 3), dtype=np.float32)
    for r, line in enumerate(rows):
        for c, samp in enumerate(cols):
            refl = np.asarray(mm[:, line, samp], dtype=np.float32)
            if not np.isfinite(refl).all() or refl.max() <= 0:
                continue
            d1, _ = _continuum_removed_depth(refl, 0.85, 1.35)
            d2, _ = _continuum_removed_depth(refl, 1.9, 2.6)
            d3, _ = _continuum_removed_depth(refl, 2.8, 3.1)
            feats[r, c] = (d1, d2, d3)
            if d3 > 0.25:
                classes[r, c] = 4          # 3 um OH/H2O feature
            elif d2 > 0.12 and d1 > 0.08:
                classes[r, c] = 0          # pyroxene-rich
            elif d1 > 0.10 and d2 <= 0.12:
                classes[r, c] = 1          # olivine-rich
            elif d1 < 0.05 and d2 < 0.06:
                classes[r, c] = 2          # feldspathic
            else:
                classes[r, c] = 3          # mixed / immature
    meta = {
        "product_id": pid,
        "grid": [int(classes.shape[0]), int(classes.shape[1])],
        "stride_lines": stride,
        "wavelength_note": "channel centers approximated as 0.8-5.0 um "
                           "near-uniform over 256 channels",
        "method": "continuum-removed band-depth heuristic (1 um, 2 um, "
                  "3 um regions); radiance-to-reflectance correction and "
                  "SAM against lab spectra are future work",
        "legend": MINERAL_LEGEND,
        "coverage_note": "computed on the IIRS native grid (~80 m/px)",
    }
    np.savez_compressed(cache_npz, classes=classes, meta=np.array(meta))
    return classes, meta


def tmc2_metric_dem(product, force=False):
    """Metric heights (m) on the 192^2 terrain grid, straight from the
    TMC-2 stereo-photogrammetry DTM GeoTIFF (true measured heights, not
    an approximation).  Extracted once; cached as
    data/reference/tmc/<pid>/metric_dem.npy.  Returns (grid, meta)."""
    pid = product["product_id"]
    cache_npy = os.path.join(TMC_CACHE, pid, "metric_dem.npy")
    if os.path.exists(cache_npy) and not force:
        g = np.load(cache_npy)
        return g, {"product_id": pid,
                   "source": "TMC-2 DTM GeoTIFF (stereo-photogrammetric, "
                             "metric)",
                   "grid_n": int(g.shape[0])}
    tif = os.path.join(TMC_CACHE, pid, "dtm.tif")
    if not os.path.exists(tif):
        ok = _extract_tif_for_product(pid, str(product.get("source", "")),
                                      tif)
        if not ok:
            return None, {"error": "DTM GeoTIFF could not be extracted "
                                   "for %s" % pid}
    return _metric_from_tif(tif, cache_npy, pid)


def _extract_tif_for_product(pid, src, out_path):
    """Two-stage stream extraction: bundle TAR -> inner product ZIP ->
    DTM GeoTIFF.  `src` may be a full path, a tar basename (resolved by
    globbing the known bundle locations), or empty (search all tars)."""
    import glob as _glob
    tar_paths = []
    if src.lower().endswith(".tar"):
        cand = [src]
        for pat in (os.path.join(ROOT, "*.tar"),
                    os.path.join(ROOT, "data", "raw", "tmc_bundles",
                                 "*.tar"),
                    os.path.join(ROOT, "data", "raw", "tmc2", "*.tar")):
            cand += [p for p in _glob.glob(pat)
                     if os.path.basename(p) == src]
        tar_paths = [c for c in cand if os.path.exists(c)]
    else:
        for pat in (os.path.join(ROOT, "*.tar"),
                    os.path.join(ROOT, "data", "raw", "tmc_bundles",
                                 "*.tar"),
                    os.path.join(ROOT, "data", "raw", "tmc2", "*.tar")):
            tar_paths += sorted(_glob.glob(pat))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    for tarp in tar_paths:
        try:
            with tarfile.open(tarp, "r:*") as t:
                for m in t:
                    if not m.name.lower().endswith(".zip"):
                        continue
                    try:
                        zf = zipfile.ZipFile(t.extractfile(m))
                        tif = next((n for n in zf.namelist()
                                    if os.path.basename(n).lower()
                                    == (pid + ".tif").lower()), None)
                        if tif is None:
                            continue
                        with zf.open(tif) as fsrc, open(tmp, "wb") as dst:
                            while True:
                                chunk = fsrc.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                dst.write(chunk)
                        os.replace(tmp, out_path)
                        return True
                    except Exception as exc:              # noqa: BLE001
                        print("layers: member %s/%s skipped: %s"
                              % (os.path.basename(tarp), m.name,
                                 str(exc)[:120]))
                        continue
        except Exception as exc:                           # noqa: BLE001
            print("layers: tar %s skipped: %s"
                  % (os.path.basename(tarp), str(exc)[:120]))
            continue
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def _dem_to_grid(dem, cache_npy, pid, native_shape):
    """Common post-processing: int meter rasters -> float, nodata-aware
    area-average downsample to the terrain grid."""
    if dem.dtype in (np.float32, np.float64, np.float16):
        dem = dem.astype(np.float32)
        dem[~np.isfinite(dem)] = np.nan
    elif dem.dtype in (np.int16, np.int32, np.uint16, np.int8, np.uint8):
        dem = dem.astype(np.float32)
        # nodata convention: extreme negative fill (-32768 for int16)
        nodata = dem <= -32000.0
        dem[nodata] = np.nan
    else:
        return None, {"error": "unsupported raster dtype %s" % dem.dtype}
    valid = np.isfinite(dem).astype(np.float32)
    filled = np.nan_to_num(dem, nan=0.0)
    v = cv2.resize(filled, (GRID_N, GRID_N), interpolation=cv2.INTER_AREA)
    w = cv2.resize(valid, (GRID_N, GRID_N), interpolation=cv2.INTER_AREA)
    g = v / np.maximum(w, 1e-6)          # nodata-aware area average
    g[w < 1e-3] = 0.0
    g = g - float(g.min())
    os.makedirs(os.path.dirname(cache_npy), exist_ok=True)
    np.save(cache_npy, g.astype(np.float32))
    return g.astype(np.float32), {"product_id": pid,
                                  "source": "TMC-2 DTM GeoTIFF "
                                            "(stereo-photogrammetric, "
                                            "metric heights in m)",
                                  "grid_n": GRID_N,
                                  "native_shape": list(map(int,
                                                           native_shape))}


def _metric_from_tif_pil(tif_path, cache_npy, pid):
    """PIL fallback for rasters exceeding OpenCV's CV_IO_MAX_IMAGE_PIXELS
    cap (the d32 DTM strips are ~1.5 GPx).  Uncompressed strips are
    STREAMED: only ~8 sampled rows per output row are ever read, so a
    2.9 GB GeoTIFF downsamples in a few MB of RAM."""
    try:
        from PIL import Image
    except ImportError:                                  # pragma: no cover
        return None, {"error": "GeoTIFF too large for OpenCV and PIL "
                               "unavailable"}
    Image.MAX_IMAGE_PIXELS = None
    try:
        im = Image.open(tif_path)
    except Exception as exc:                             # noqa: BLE001
        return None, {"error": "GeoTIFF not readable (PIL open failed: %s)"
                               % str(exc)[:110]}
    try:
        w, h = im.size
        tag = getattr(im, "tag_v2", {})
        bits = tag.get(258, 16)
        bits = bits[0] if isinstance(bits, (tuple, list)) else bits
        fmt = tag.get(339, 1)
        fmt = fmt[0] if isinstance(fmt, (tuple, list)) else fmt
        np_dt = {(16, 2): np.int16, (16, 1): np.uint16,
                 (32, 2): np.int32, (32, 3): np.float32,
                 (32, 1): np.uint32, (8, 1): np.uint8}.get((int(bits),
                                                            int(fmt)))
        if np_dt is None:
            return None, {"error": "unsupported TIFF geometry "
                                   "(bits=%s fmt=%s)" % (bits, fmt)}
        comp = tag.get(259, 1)
        comp = comp[0] if isinstance(comp, (tuple, list)) else comp
        if int(comp) != 1:
            # compressed: one full decode, then the common path
            arr = np.asarray(im).astype(np.float32)
            return _dem_to_grid(arr, cache_npy, pid, (h, w))
        offsets, counts = tag.get(273), tag.get(279)
        if offsets is None:
            return None, {"error": "TIFF strip table missing"}
        if not isinstance(offsets, (tuple, list)):
            offsets = [offsets]
        rps = tag.get(278, 1)
        rps = int(rps[0] if isinstance(rps, (tuple, list)) else rps) or 1
        row_bytes = w * np.dtype(np_dt).itemsize
        step = max(1, w // GRID_N)
        n_samp = max(1, min(8, h // GRID_N))
        is_float = np_dt == np.float32
        sums = np.zeros((GRID_N, GRID_N), np.float64)
        wgts = np.zeros((GRID_N, GRID_N), np.float64)
        fp = im.fp
        for gy in range(GRID_N):
            y0 = h * gy // GRID_N
            y1 = max(h * (gy + 1) // GRID_N, y0 + 1)
            ys = np.unique(np.linspace(y0, y1 - 1,
                                       min(n_samp, y1 - y0)).astype(int))
            for y in ys:
                si = int(y) // rps
                fp.seek(int(offsets[si]) + (int(y) % rps) * row_bytes)
                row = np.frombuffer(fp.read(row_bytes),
                                    dtype=np_dt).astype(np.float32)
                if len(row) < w:
                    continue
                if is_float:
                    row[~np.isfinite(row)] = np.nan
                else:
                    row[row <= -32000.0] = np.nan
                use = row[:GRID_N * step].reshape(GRID_N, step)
                sums += np.nan_to_num(use, nan=0.0).sum(axis=1)
                wgts += np.isfinite(use).sum(axis=1)
        g = sums / np.maximum(wgts, 1e-6)
        g[wgts < 1e-3] = 0.0
        g = g - float(g.min())
        os.makedirs(os.path.dirname(cache_npy), exist_ok=True)
        np.save(cache_npy, g.astype(np.float32))
        return g.astype(np.float32), {
            "product_id": pid,
            "source": "TMC-2 DTM GeoTIFF (stereo-photogrammetric, metric "
                      "heights in m) [streamed PIL reader]",
            "grid_n": GRID_N,
            "native_shape": [int(h), int(w)]}
    finally:
        try:
            im.close()
        except Exception:                                # noqa: BLE001
            pass


def _metric_from_tif(tif_path, cache_npy, pid):
    """Read the DTM GeoTIFF as metric heights (m) and downsample to the
    terrain grid.  Handles float32/float64 rasters and int16/int32
    meter-encoded rasters (PRADAN TMC-2 DTMs ship int16 m with -32768 as
    nodata).  Falls back to a streaming PIL reader for rasters bigger
    than OpenCV's pixel cap, then fails honestly."""
    try:
        dem = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    except Exception:                                    # noqa: BLE001
        dem = None
    if dem is not None:
        return _dem_to_grid(dem, cache_npy, pid, dem.shape)
    return _metric_from_tif_pil(tif_path, cache_npy, pid)


def layers_payload(scene_dir, meta):
    """Per-scene layer availability block for the 02 TERRAIN 3D layer
    switcher.  Honest by construction: a layer is 'available' only when
    real data supports it for THIS scene."""
    import tmc as tmc_mod
    import iirs as iirs_mod
    rel = os.path.relpath(scene_dir, PROC).replace("\\", "/")
    out = {
        "height_sfs": {"available": True,
                       "note": "linearized-Lambertian shape-from-shading "
                               "(photometric approximation, non-metric)"},
        "optical_texture": {"available": os.path.exists(
                                os.path.join(scene_dir, "source.png")),
                            "url": "/static/%s/source.png" % rel,
                            "note": "source radiance image draped on "
                                    "the mesh"},
        "metric_dem_tmc2": {"available": False,
                            "note": "no TMC-2 stereo DTM co-registered on "
                                    "this scene's ground yet"},
        "minerals_iirs": {"available": False,
                          "note": "no IIRS cube co-registered on this "
                                  "scene's ground yet"},
        "sfs_vs_metric": {"available": False,
                          "note": "becomes available once a TMC-2 metric "
                                  "DEM covers this scene (measured "
                                  "accuracy map of the SFS relief)"},
        "legend_iirs": MINERAL_LEGEND,
    }
    # Prefer the scene's ALREADY-SELECTED cross-instrument references
    # (recorded at ingest time by the auto-selection in ingest.py) - a
    # scene's own product_id only matches its own instrument, so the
    # substring fallback below is just a safety net.
    def _pick(lib_products, ref_key):
        ref_pid = ((meta or {}).get(ref_key) or {}).get("product_id")
        if ref_pid:
            p = next((q for q in lib_products
                      if q["product_id"] == ref_pid), None)
            if p is not None:
                return p
        pid = (meta or {}).get("product_id", "")
        return next((q for q in lib_products
                     if pid and (q["product_id"] in pid or
                                 pid in q["product_id"])), None)

    tmc_prod = _pick(tmc_mod.all_products(), "tmc_reference")
    if tmc_prod and "dtm" in (tmc_prod.get("product_kind") or "").lower():
        g, m = tmc2_metric_dem(tmc_prod)
        out["metric_dem_tmc2"] = {
            "available": g is not None,
            "product_id": tmc_prod["product_id"],
            "grid": (g.round(2).tolist() if g is not None else None),
            "note": m.get("source", "") if g is not None
                    else m.get("error", ""),
        }
    iirs_prod = _pick(iirs_mod.all_products(), "iirs_reference")
    if iirs_prod:
        classes, m = iirs_minerals(iirs_prod)
        if classes is not None:
            cg = cv2.resize(classes, (GRID_N, GRID_N),
                            interpolation=cv2.INTER_NEAREST)
            out["minerals_iirs"] = {
                "available": True,
                "product_id": iirs_prod["product_id"],
                "classes": cg.tolist(),
                "legend": MINERAL_LEGEND,
                "method": m.get("method"),
                "coverage_note": m.get("coverage_note"),
            }
        else:
            out["minerals_iirs"] = {
                "available": False,
                "product_id": iirs_prod["product_id"],
                "error": m.get("error", "classification unavailable")}
    if out["metric_dem_tmc2"]["available"] and \
            out["metric_dem_tmc2"].get("grid"):
        dem_path = os.path.join(scene_dir, "dem.npy")
        if os.path.exists(dem_path):
            sfs = np.load(dem_path).astype(np.float32)
            sfs = cv2.resize(sfs, (GRID_N, GRID_N),
                             interpolation=cv2.INTER_AREA)
            sfs -= sfs.min()
            met = np.array(out["metric_dem_tmc2"]["grid"],
                           dtype=np.float32)
            met -= met.min()
            if sfs.max() > 0 and met.max() > 0:
                # honest validation: both fields normalized to [0,1] -
                # SFS is RELATIVE local relief while the metric DTM carries
                # ABSOLUTE strip-wide topography, so raw meter differences
                # are meaningless; shape agreement (Pearson r) + normalized
                # MAE are the defensible statistics
                s01 = sfs / sfs.max()
                m01 = met / met.max()
                diff = s01 - m01
                r = float(np.corrcoef(s01.ravel(), m01.ravel())[0, 1])
                out["sfs_vs_metric"] = {
                    "available": True,
                    "pearson_r": round(r, 3),
                    "mae_norm": round(float(np.abs(diff).mean()), 4),
                    "max_dev_norm": round(float(np.abs(diff).max()), 4),
                    "note": "shape-from-shading vs TMC-2 stereo metric "
                            "DEM over the same ground, both normalized "
                            "to [0,1] (SFS is relative relief, the DTM "
                            "is absolute height - Pearson r measures "
                            "shape agreement; r near 0 is expected and "
                            "honest for shaded-relief input)",
                }
    return out
