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


# ---------------------------------------------------------------- geography
def scene_geo(meta):
    """(center {lat_deg, lon_deg}, extent_m) for a scene, or (None, None).
    Center comes from source_footprint_center (the crop's own selenographic
    position); extent from the analysis grid."""
    meta = meta or {}
    c = meta.get("source_footprint_center") or {}
    if not c.get("lat_deg") or not c.get("lon_deg"):
        return None, None
    g = meta.get("analysis_grid") or {}
    extent = float(g.get("cell_meters", 1.0)) * float(g.get("n", 1024))
    return c, extent


def _wrap_lon(lon):
    return lon % 360.0


def geo_pixel_window(corners, H, W, center, extent_m, gsd_m):
    """Pixel window (r0, c0, r1, c1) in an HxW raster for a scene at
    `center`, using the product's footprint corners as the georeference
    (inverse-bilinear solve - polar-swath and lon-wrap safe).  Returns
    None when the center falls clearly outside the footprint (no
    geographic coverage - honest no-data, never a wrong-region view)."""
    import tmc as _tmc
    uv = _tmc.geo_uv(corners, center)
    if uv is None or not H or not W:
        return None
    fx, fy = uv
    px = fx * (W - 1)
    py = fy * (H - 1)
    half = max(2.0, extent_m / 2.0 / float(gsd_m))
    r0 = int(max(0, py - half))
    r1 = int(min(H, py + half))
    c0 = int(max(0, px - half))
    c1 = int(min(W, px + half))
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    return r0, c0, r1, c1


def _dtm_gsd_m(product):
    """TMC-2 DTM ground sample distance from the product id (d18 = 18 m/px,
    d32 = 32 m/px); 24 m fallback."""
    pid = (product.get("product_id") or "")
    m = re.search(r"_d(\d+)(?:_\D*)?$", pid)
    return float(m.group(1)) if m else 24.0


def _cache_key(center):
    """Cache suffix bound to the scene's geographic position, so windows
    for different crops of the same product never collide."""
    return "%d_%d" % (round(float(center["lat_deg"]) * 1000),
                      round(float(center["lon_deg"]) * 1000))

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


def iirs_minerals(product, stride=64, force=False, center=None,
                  extent_m=None):
    """Classified mineral map for an IIRS product (classes HxW uint8 +
    meta), cached to data/reference/iirs/<pid>/minerals*.npz.  When the
    scene's geographic center is given, ONLY the window of the cube that
    actually covers the scene (footprint-corner georeferencing, ~80 m/px)
    is classified - never a wrong-region view; no coverage is an honest
    error, not somebody else's terrain."""
    pid = product["product_id"]
    key = _cache_key(center) if center else None
    cache_npz = os.path.join(IIRS_CACHE, pid,
                             ("minerals_%s.npz" % key) if key
                             else "minerals.npz")
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
        # last resort: pull the product zip from the public Kaggle
        # dataset (remote hosting) and stream the cube out of it
        try:
            import kfetch
            kfetch.ensure_cube(product, cube)
        except Exception as exc:                         # noqa: BLE001
            print("layers: kfetch cube:", str(exc)[:120])
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
    # geographic window (IIRS nominal ~80 m/px) or legacy whole-cube stride
    win = None
    gsd = 80.0
    if center:
        win = geo_pixel_window(
            (product.get("geometry") or {}).get("footprint_corners"),
            lines, samples, center, extent_m or 1024.0, gsd)
        if win is None:
            return None, {"error": "IIRS cube %s does not cover the "
                                   "scene's coordinates (lat %.2f, lon %.2f)"
                                   % (pid, float(center["lat_deg"]),
                                      _wrap_lon(float(center["lon_deg"])))}
    if win:
        r0, c0, r1, c1 = win
        rows = list(range(r0, r1)) if r1 - r0 <= 160 else \
            list(range(r0, r1, max(1, (r1 - r0) // 128)))
        cols = list(range(c0, c1)) if c1 - c0 <= 160 else \
            list(range(c0, c1, max(1, (c1 - c0) // 128)))
    else:
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
        "geographic_window": ({"center": center,
                               "pixel_window": [int(v) for v in win],
                               "gsd_m": gsd,
                               "note": "classified ONLY the cube window "
                                       "covering the scene's coordinates"}
                              if (center and win) else None),
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


def tmc2_metric_dem(product, force=False, center=None, extent_m=None):
    """Metric heights (m) on the 192^2 terrain grid from the TMC-2 stereo-
    photogrammetry DTM GeoTIFF (true measured heights, not an approximation).
    When the scene's geographic center is given, ONLY the DTM window that
    actually covers the scene (footprint-corner georeferencing, gsd from
    the product id: d18 = 18 m/px, d32 = 32 m/px) is extracted - the layer
    renders the scene's own terrain at the scene's own scale, never a
    wrong-region or wrong-scale view.  Cached per (product, scene center)."""
    pid = product["product_id"]
    key = _cache_key(center) if center else None
    cache_npy = os.path.join(TMC_CACHE, pid,
                             ("metric_dem_%s.npy" % key) if key
                             else "metric_dem.npy")
    if os.path.exists(cache_npy) and not force:
        g = np.load(cache_npy)
        side = cache_npy + ".meta.json"
        extra = json.load(open(side)) if os.path.exists(side) else {}
        return g, {"product_id": pid,
                   "source": "TMC-2 DTM GeoTIFF (stereo-photogrammetric, "
                             "metric)",
                   "grid_n": int(g.shape[0]),
                   "geographic_window": extra.get("geographic_window")}
    tif = os.path.join(TMC_CACHE, pid, "dtm.tif")
    if not os.path.exists(tif):
        ok = _extract_tif_for_product(pid, str(product.get("source", "")),
                                      tif)
        if not ok:
            # last resort: pull the product zip from the public Kaggle
            # dataset (remote hosting) and stream the DTM out of it
            try:
                import kfetch
                ok = kfetch.ensure_dtm(product, tif)
            except Exception as exc:                     # noqa: BLE001
                print("layers: kfetch dtm:", str(exc)[:120])
        if not ok:
            return None, {"error": "DTM GeoTIFF could not be extracted "
                                   "for %s" % pid}
    if center:
        corners = (product.get("geometry") or {}).get("footprint_corners")
        gsd = _dtm_gsd_m(product)
        g, m = _metric_window_from_tif(tif, corners, center,
                                       extent_m or 1024.0, gsd,
                                       cache_npy, pid)
        if g is None and "error" not in m:
            # fall back to the whole-raster read (no corners available)
            return _metric_from_tif(tif, cache_npy, pid)
        return g, m
    return _metric_from_tif(tif, cache_npy, pid)


def _read_tif_window(tif_path, r0, r1, c0, c1):
    """Low-memory window read of a (potentially multi-GB) GeoTIFF.

    Strategy: tifffile memmap (zero-copy virtual memory) -> tifffile
    offset read (uncontiguous files: read only the window's row bytes) ->
    PIL last resort (loads the full page - only acceptable for small
    rasters).  Never decodes a whole gigapixel page into RAM.
    Returns (window_array_2d, dtype_str) or (None, error_str)."""
    try:
        import tifffile
    except ImportError:
        tifffile = None
    if tifffile is not None:
        try:
            mm = tifffile.memmap(tif_path, mode="r")
            win = np.asarray(mm[r0:r1, c0:c1])
            del mm
            return win, str(win.dtype)
        except Exception:                                # noqa: BLE001
            pass
        try:
            with tifffile.TiffFile(tif_path) as tif:
                pg = tif.pages[0]
                h, w = int(pg.shape[0]), int(pg.shape[1])
                dt = np.dtype(pg.dtype)
                itemsize = dt.itemsize
                rowbytes = w * itemsize
                r0c, r1c = max(0, int(r0)), min(int(r1), h)
                c0c, c1c = max(0, int(c0)), min(int(c1), w)
                if getattr(pg, "is_contiguous", False):
                    ob = int(pg.dataoffsets[0])
                    with open(tif_path, "rb") as f:
                        f.seek(ob + r0c * rowbytes)
                        raw = f.read((r1c - r0c) * rowbytes)
                    arr = np.frombuffer(raw, dtype=dt,
                                        count=(r1c - r0c) * w)
                    arr = arr.reshape(r1c - r0c, w)[:, c0c:c1c].copy()
                    return arr, str(dt)
                # stripped layout: read ONLY the strips intersecting the
                # window (each strip is a few hundred KB - RAM-safe)
                rps = int(getattr(pg, "rowsperstrip", 0) or 1)
                rps = max(1, min(rps, h))
                offs = np.atleast_1d(pg.dataoffsets)
                cnts = np.atleast_1d(pg.databytecounts)
                s0 = r0c // rps
                s1 = (r1c - 1) // rps
                chunks = []
                with open(tif_path, "rb") as f:
                    for s in range(s0, s1 + 1):
                        rows_in = min(rps, h - s * rps)
                        f.seek(int(offs[s]))
                        raw = f.read(int(cnts[s]))
                        strip = np.frombuffer(raw, dtype=dt)
                        strip = strip[:rows_in * w].reshape(rows_in, w)
                        chunks.append(strip)
                arr = np.concatenate(chunks, axis=0)[
                    r0c - s0 * rps: r1c - s0 * rps, c0c:c1c].copy()
                return arr, str(dt)
        except Exception:                                # noqa: BLE001
            pass
    # PIL fallback: only sane for modest rasters; guard the decode size
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(tif_path)
    w, h = im.size
    if w * h > 400_000_000:                              # >400 MP: refuse
        im.close()
        return None, ("raster too large for the PIL fallback (%d MP) - "
                      "install tifffile for low-memory window reads"
                      % (w * h // 1_000_000))
    arr = np.asarray(im.crop((c0, r0, c1, r1)))
    im.close()
    return arr, str(arr.dtype)


def _metric_window_from_tif(tif_path, corners, center, extent_m, gsd_m,
                            cache_npy, pid):
    """Geographic window extraction.  Returns (grid, meta); meta carries
    'no_coverage' when the DTM simply does not cover the scene."""
    try:
        with open(tif_path, "rb") as _f:
            pass                                          # existence probe
    except Exception as exc:                             # noqa: BLE001
        return None, {"error": "GeoTIFF not readable (%s)" % str(exc)[:110]}
    im_h = im_w = None
    try:
        import tifffile
        with tifffile.TiffFile(tif_path) as tif:
            pg = tif.pages[0]
            im_h, im_w = int(pg.shape[0]), int(pg.shape[1])
    except Exception:                                    # noqa: BLE001
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(tif_path)
        im_w, im_h = im.size
        im.close()
    win = geo_pixel_window(corners, im_h, im_w, center, extent_m, gsd_m)
    if win is None:
        return None, {"no_coverage": True,
                      "error": "TMC-2 DTM %s does not cover the scene's "
                               "coordinates (lat %.2f, lon %.2f)"
                               % (pid, float(center["lat_deg"]),
                                  _wrap_lon(float(center["lon_deg"])))}
    r0, c0, r1, c1 = win
    arr, dt = _read_tif_window(tif_path, r0, r1, c0, c1)
    if arr is None:
        return None, {"error": "GeoTIFF window read failed: %s" % dt}
    g, m = _dem_to_grid(arr, cache_npy, pid, arr.shape)
    if g is not None:
        m["geographic_window"] = {
            "center": center,
            "pixel_window": [int(v) for v in win],
            "gsd_m": gsd_m,
            "native_window_m": [(r1 - r0) * gsd_m, (c1 - c0) * gsd_m],
            "note": "extracted ONLY the DTM window covering the scene's "
                    "coordinates - true metric relief at scene scale",
        }
        try:
            with open(cache_npy + ".meta.json", "w") as f:
                json.dump({"geographic_window": m["geographic_window"]}, f)
        except Exception:                                # noqa: BLE001
            pass
    return g, m


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
    resample to the terrain grid.  Downsampling uses area-averaging;
    upsampling (a polar window smaller than the grid) uses bicubic plus a
    light smoothing pass so the METRIC layer renders as continuous
    terrain instead of terraced blocks - interpolation only, no
    fabricated values beyond the sampled window."""
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
    upsample = min(dem.shape) < GRID_N
    interp = cv2.INTER_CUBIC if upsample else cv2.INTER_AREA
    v = cv2.resize(filled, (GRID_N, GRID_N), interpolation=interp)
    w = cv2.resize(valid, (GRID_N, GRID_N), interpolation=cv2.INTER_AREA)
    g = v / np.maximum(w, 1e-6)          # nodata-aware area average
    g[w < 1e-3] = 0.0
    if upsample:
        # light smoothing on the upsampled path only (kills bicubic
        # ringing / pixel terracing; sub-kilometre scale)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        g[w < 1e-3] = 0.0
    g = g - float(g.min())
    # honest guard: a window with no valid data (or zero relief) must not
    # render as a fake metric surface
    if not np.isfinite(g).all() or float(g.std()) < 1e-6 or \
            (valid.sum() if valid.ndim else 0) == 0:
        return None, {"error": "TMC-2 DTM window contains no valid data "
                               "at these coordinates (polar coverage gap)",
                      "no_valid_data": True}
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
    center, extent_m = scene_geo(meta)
    tmc_ref_meta = (meta or {}).get("tmc_reference") or {}
    if tmc_ref_meta.get("status") not in (None, "", "selected"):
        # selection already ruled this region out (no_coverage etc.) -
        # surface that reason instead of attempting unrelated extraction
        out["metric_dem_tmc2"]["note"] = tmc_ref_meta.get("note") or \
            "nearest TMC-2 DTM does not cover this scene's coordinates"
    elif tmc_prod and "dtm" in (tmc_prod.get("product_kind") or "").lower():
        g, m = tmc2_metric_dem(tmc_prod, center=center, extent_m=extent_m)
        out["metric_dem_tmc2"] = {
            "available": g is not None,
            "product_id": tmc_prod["product_id"],
            "grid": (g.round(2).tolist() if g is not None else None),
            "center": center,
            "geographic_window": m.get("geographic_window"),
            "note": (m.get("source", "") if g is not None
                     else m.get("error", "")),
        }
    iirs_prod = _pick(iirs_mod.all_products(), "iirs_reference")
    iirs_ref_meta = (meta or {}).get("iirs_reference") or {}
    if iirs_ref_meta.get("status") not in (None, "", "selected"):
        out["minerals_iirs"]["note"] = iirs_ref_meta.get("note") or \
            "nearest IIRS cube does not cover this scene's coordinates"
    elif iirs_prod:
        classes, m = iirs_minerals(iirs_prod, center=center,
                                   extent_m=extent_m)
        if classes is not None:
            # smooth-boundary upsample: one-hot linear interpolation then
            # argmax keeps every class inside the legend (no invented ids)
            # while removing hard native-resolution blocks
            onehot = np.zeros((int(classes.max()) + 1,) + classes.shape,
                              dtype=np.float32)
            for cid in range(onehot.shape[0]):
                onehot[cid] = (classes == cid).astype(np.float32)
            up = np.stack([cv2.resize(onehot[c], (GRID_N, GRID_N),
                                      interpolation=cv2.INTER_LINEAR)
                           for c in range(onehot.shape[0])])
            cg = up.argmax(axis=0).astype(np.uint8)
            out["minerals_iirs"] = {
                "available": True,
                "product_id": iirs_prod["product_id"],
                "classes": cg.tolist(),
                "legend": MINERAL_LEGEND,
                "method": m.get("method"),
                "coverage_note": m.get("coverage_note"),
                "center": center,
                "geographic_window": m.get("geographic_window"),
            }
        else:
            out["minerals_iirs"] = {
                "available": False,
                "product_id": iirs_prod["product_id"],
                "center": center,
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
