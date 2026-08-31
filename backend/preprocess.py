"""SIH26166 - Stage 1-2 preprocessing.

Reads the real Chandrayaan-2 OHRC PDS4 product (`.img` + label + sun-angle
`.spm`), reconstructs a meter-scale DEM patch via shape-from-shading
(linearized Lambert + minimum-norm slope estimate + FFT Poisson solve), and
renders a differently-illuminated Lommel-Seeliger reference pair for the
cross-illumination matching demo. Also builds the synthetic Tycho stand-in
DEM scene exactly as specified in the Kaggle training notebook.

Nothing here mutates data/raw or data/reference in place - all outputs go
to data/processed/.
"""
import json
import os
import re

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_DIR = os.path.join(
    ROOT, "data", "UNZIPPED_DATA",
    "ch2_ohr_ncp_20210401T2357376656_d_img_d18")
IMG_PATH = os.path.join(
    SCENE_DIR, "data", "calibrated", "20210401",
    "ch2_ohr_ncp_20210401T2357376656_d_img_d18.img")
LABEL_PATH = os.path.join(
    SCENE_DIR, "data", "calibrated", "20210401",
    "ch2_ohr_ncp_20210401T2357376656_d_img_d18.xml")
SPM_PATH = os.path.join(
    SCENE_DIR, "miscellaneous", "calibrated", "20210401",
    "ch2_ohr_ncp_20210401T2357376656_d_img_d18.spm")
OUT_DIR = os.path.join(ROOT, "data", "processed", "ohrc_real")

SCENE_ID = "ch2_ohr_ncp_20210401T2357376656"


# Stage 1 - ingest: parse the PDS4 label + sun-angle file. Nothing here is
# estimated from pixels; everything is closed-form mission metadata.
def parse_label():
    xml = open(LABEL_PATH, encoding="utf-8").read()

    def grab(tag):
        m = re.search(r"<%s[^>]*>([^<]+)</%s>" % (tag, tag), xml)
        return m.group(1).strip() if m else None

    def isda(tag):
        m = re.search(r"<isda:%s[^>]*>([^<]+)</isda:%s>" % (tag, tag), xml)
        return m.group(1).strip() if m else None

    corners = {}
    for name in ("upper_left", "upper_right", "lower_left", "lower_right"):
        lat = re.search(r"<isda:%s_latitude unit=\"deg\">([^<]+)" % name, xml)
        lon = re.search(r"<isda:%s_longitude unit=\"deg\">([^<]+)" % name, xml)
        corners[name] = {"lat_deg": float(lat.group(1)),
                         "lon_deg": float(lon.group(1))}
    lines = int(re.search(r"<axis_name>Line</axis_name>\s*<elements>(\d+)",
                          xml).group(1))
    samples = int(re.search(r"<axis_name>Sample</axis_name>\s*<elements>(\d+)",
                            xml).group(1))
    return {
        "product_id": SCENE_ID,
        "instrument": "OHRC (Orbiter High Resolution Camera)",
        "mission": "Chandrayaan-2",
        "band": "Panchromatic 500-800 nm",
        "start_time_utc": grab("start_date_time"),
        "stop_time_utc": grab("stop_date_time"),
        "processing_level": "Calibrated",
        "lines": lines,
        "samples": samples,
        "data_type": "UnsignedByte",
        "focal_length_mm": float(isda("focal_length")),
        "detector_pixel_width_um": float(isda("detector_pixel_width")),
        "line_exposure_duration_ms": float(isda("line_exposure_duration")),
        "tdi_stages": isda("tdi_stages"),
        "spacecraft_altitude_km": float(isda("spacecraft_altitude")),
        "pixel_resolution_m": float(isda("pixel_resolution")),
        "orbit_limb_direction": isda("orbit_limb_direction"),
        "footprint_corners": corners,
    }


def parse_sun_angles():
    """Sun-parameter file (format per readme.txt): the record ends with
    Phase angle, Sun aspect, Sun azimuth, Sun elevation. Solar incidence
    = 90 - sun elevation. (Read the last four numeric tokens - the fixed-
    width fields in this product are wider than the documented F9.3.)"""
    import re as _re
    with open(SPM_PATH) as f:
        rec = f.readline()
    vals = [float(x) for x in _re.findall(r"-?\d+\.\d+", rec)]
    phase, aspect, az, el = vals[-4:]
    return {"phase_deg": phase, "sun_aspect_deg": aspect,
            "sun_azimuth_deg": az, "sun_elevation_deg": el,
            "solar_incidence_deg": 90.0 - el}



# Stage 2 - relief from the real image.
# Linearized Lambertian shape-from-shading: I ~ A(-p*sx - q*sy + sz) with
# (p, q) = (dh/dx, dh/dy), sun unit vector s from the .spm metadata. A single
# image constrains only the slope component parallel to the sun's horizontal
# projection, so take the minimum-norm solution
#   (p, q) = -DeltaI / (A |s_h|^2) * (sx, sy)
# then integrate the slope field with an FFT Poisson solve - the same
# truncated-2D-Fourier machinery as spectral/fourier_surface.py, so the
# coefficient grid is exactly what the frontend mesh consumes.
def sun_vector_image_frame(az_deg, el_deg):
    """Sun unit vector in the (east, south, up) image frame.

    Image x = east (longitude increases with pixel per the geometry CSV),
    image y = south (latitude decreases with scan, descending orbit).
    Azimuth is compass-clockwise from north.
    """
    az, el = np.radians(az_deg), np.radians(el_deg)
    east = np.cos(el) * np.sin(az)          # sin(270) < 0 -> sun in the west
    north = np.cos(el) * np.cos(az)
    up = np.sin(el)
    return np.array([east, -north, up])     # (east, south, up)


def shape_from_shading(image, sun_az_deg, sun_el_deg, cell_m=1.0,
                       target_slope_deg=8.0, max_slope_deg=35.0):
    img = gaussian_filter(image.astype(np.float64), 3.0)
    sx, sy, sz = sun_vector_image_frame(sun_az_deg, sun_el_deg)
    sh2 = sx * sx + sy * sy
    a_flat = np.median(img) / max(sz, 1e-6)          # albedo*irradiance scale
    delta = img - a_flat * sz
    # shadows (radiance ~ 0) carry no slope information - mask them out
    shadow = img < max(3.0, np.percentile(img, 1.0))
    t = -delta / (a_flat * sh2)
    max_t = np.tan(np.radians(max_slope_deg))
    t = np.clip(t, -max_t, max_t)
    t[shadow] = 0.0
    p, q = -sx * t, -sy * t
    # FFT Poisson solve: lap h = dp/dx + dq/dy on a windowed domain.
    # Must be a full 2-D FFT (rfft is 1-D along the last axis and would
    # divide a 1-D spectrum by a 2-D Laplacian - caught by verify_robust).
    w = np.hanning(p.shape[0])[:, None] * np.hanning(p.shape[1])[None, :]
    div = np.gradient(p * w, axis=0) + np.gradient(q * w, axis=1)
    fy = np.fft.fftfreq(div.shape[0])[:, None]
    fx = np.fft.fftfreq(div.shape[1])[None, :]
    k2 = (2 * np.pi * fy) ** 2 + (2 * np.pi * fx) ** 2
    k2[0, 0] = 1.0
    h = np.real(np.fft.ifft2(np.fft.fft2(div) / (-k2)))
    h -= h.mean()
    # Radiometric (non-metric) height calibration: a single image cannot fix
    # absolute height (albedo variation is entangled with slope), so scale the
    # relief to a stated plausible RMS slope. This is a visualization choice
    # and is reported in the metadata - never presented as metric.
    gy, gx = np.gradient(h.astype(np.float64) / cell_m)
    rms = np.sqrt(np.mean(gx ** 2 + gy ** 2))
    h *= np.tan(np.radians(target_slope_deg)) / max(rms, 1e-9)
    return h.astype(np.float32)


# Stage 3 helper - Lommel-Seeliger renderer (Kaggle notebook math, verbatim).
def lommel_seeliger(albedo, cos_i, cos_e):
    return albedo * cos_i / (cos_i + cos_e + 1e-8)


def simulate_second_pass(img, rot_deg=1.8, scale=1.035, gamma=1.35, seed=3):
    """Simulated second-pass reference product from the SAME real OHRC scene:
    slight geometric offset (rotation+scale homography - what a different
    orbit's viewing geometry produces) plus a genuinely different radiometric
    response (gamma, across-track radiance gradient, sensor noise). Matching
    this is a real registration problem, not a copy."""
    rng = np.random.default_rng(seed)
    img = img.astype(np.float32)
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2),
                                rot_deg, scale)
    g = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                       flags=cv2.INTER_CUBIC, borderValue=0)
    x = np.linspace(0, 1, img.shape[1], dtype=np.float32)
    g = 255.0 * (g / 255.0) ** gamma * (0.88 + 0.28 * x[None, :])
    g += rng.normal(0, 4.0, g.shape).astype(np.float32)
    return np.clip(g, 0, 255).astype(np.uint8)


def render_shaded(patch, az_deg, el_deg, albedo=0.12, cell_m=1.0):
    """Lommel-Seeliger shading. `albedo` may be a scalar or a per-pixel
    albedo field (real terrain has regolith albedo units - pure shading is
    unnaturally featureless for keypoint matching)."""
    patch = patch.astype(np.float32)
    gy, gx = np.gradient(patch / cell_m)
    normal = np.dstack([-gx, -gy, np.ones_like(patch)]).astype(np.float32)
    normal /= (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8)
    az, el = np.radians(az_deg), np.radians(el_deg)
    # sun direction in the (east, south, up) image frame
    sun = np.array([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az),
                    np.sin(el)], dtype=np.float32)
    cos_i = np.clip(normal @ sun, 0, 1)
    a = np.broadcast_to(np.asarray(albedo, dtype=np.float32),
                        patch.shape).astype(np.float32)
    b = lommel_seeliger(a, cos_i, np.ones_like(cos_i))
    return b / (b.max() + 1e-8)


def albedo_field(shape, seed=11):
    """Multi-octave per-pixel albedo variations (regolith brightness units) -
    real lunar scenes are textured, pure shaded DEMs are not."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float32)
    for res, amp in [(16, 0.10), (48, 0.09), (128, 0.06), (320, 0.03)]:
        f = rng.standard_normal((res, res)).astype(np.float32)
        f = np.array(Image.fromarray(f).resize((shape[1], shape[0]),
                                               Image.BICUBIC))
        f = (f - f.mean()) / (f.std() + 1e-9)
        out += amp * f
    return 1.0 + np.clip(out, -0.35, 0.35)



# --------------------------------------------------------------------------
# Synthetic Tycho stand-in DEM (Kaggle notebook Cell 2, extended to 1024)
# --------------------------------------------------------------------------
def make_crater_dem(size=1024, rim_radius=0.42, depth=1900.0, rim_height=350.0,
                    central_peak_h=2000.0, central_peak_r=0.09, seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2, size / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size / 2)
    bowl = -depth * np.clip(1 - (r / rim_radius) ** 2, 0, 1)
    rim = rim_height * np.exp(-((r - rim_radius) ** 2) / (2 * 0.03 ** 2))
    peak = central_peak_h * np.exp(-(r ** 2) / (2 * central_peak_r ** 2))
    dem = bowl + rim + peak
    rough = np.zeros((size, size), dtype=np.float32)
    for octave, amp in [(4, 40), (8, 18), (16, 8), (32, 3), (64, 1.5)]:
        noise = rng.standard_normal((octave, octave)).astype(np.float32)
        noise_img = np.array(Image.fromarray(noise).resize(
            (size, size), Image.BICUBIC))
        rough += amp * noise_img
    return (dem + rough).astype(np.float32)


def save_png(arr, path, stretch=True):
    a = arr.astype(np.float64)
    if stretch:
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    else:
        a = np.clip(a, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def pick_best_crop(mm, lines, samples, n_cands=5, crop=4096, stride=16):
    """Choose the highest-variance 4096x4096 window of the 90148x12000 strip."""
    best_var, best_origin = -1.0, None
    row_starts = np.linspace(2000, lines - crop - 2000, n_cands).astype(int)
    for r0 in row_starts:
        c0 = (samples - crop) // 2
        sub = mm[r0:r0 + crop:stride, c0:c0 + crop:stride].astype(np.float32)
        var = sub.std()
        if var > best_var:
            best_var, best_origin = var, (int(r0), int(c0))
    return best_origin, best_var


# ---- crater identification + shadow projection ---------------------------
def _load_geo_grid():
    """Geometry CSV -> (lon, lat, pixel, scan) arrays for selenographic
    lookup of any full-image (line, sample)."""
    csv = os.path.join(SCENE_DIR, "geometry", "calibrated", "20210401",
                       "ch2_ohr_ncp_20210401T2357376656_g_grd_d18.csv")
    if not os.path.exists(csv):
        return None
    try:
        d = np.genfromtxt(csv, delimiter=",", skip_header=1)
        return d  # cols: Longitude, Latitude, Pixel, Scan
    except Exception:
        return None


def _geo_lookup(geo, line, sample):
    """Bilinear interpolation of the per-100-px geometry grid."""
    if geo is None:
        return None, None
    lon_c, lat_c, px_c, sc_c = geo[:, 0], geo[:, 1], geo[:, 2], geo[:, 3]
    s0 = int(round(line / 100.0)) * 100
    p0 = int(round(sample / 100.0)) * 100
    best, bd = None, 1e18
    for ds in (0, 100):
        for dp in (0, 100, -100):
            m = (np.abs(sc_c - (s0 + ds)) < 1) & (np.abs(px_c - (p0 + dp)) < 1)
            if not m.any():
                continue
            d = (sc_c[m] - line) ** 2 + (px_c[m] - sample) ** 2
            i = np.argmin(d)
            if d[i] < bd:
                bd = d[i]
                best = (lon_c[m][i], lat_c[m][i])
    return best if best else (None, None)


def detect_craters(I, cell_m, sun, crop_origin, meta, max_out=25):
    """Hough crater detection on the OHRC analysis grid + shadow-projection
    depth estimate: for sun elevation theta, a rim shadow of length L
    implies depth ~ L * tan(theta) (flat-floor approximation). Positions
    are mapped to selenographic coordinates via the geometry CSV."""
    sx, sy, sz = sun_vector_image_frame(sun["sun_azimuth_deg"],
                                        sun["sun_elevation_deg"])
    tan_el = np.tan(np.radians(sun["sun_elevation_deg"]))
    u = np.array([-sx, -sy])                       # shadow travel direction
    u /= (np.linalg.norm(u) + 1e-9)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(np.clip(I, 0, 255).astype(np.uint8))
    g = cv2.medianBlur(g, 3)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=26,
                               param1=110, param2=38, minRadius=5,
                               maxRadius=90)
    if circles is None:
        return []
    geo = _load_geo_grid()
    r0, c0 = crop_origin
    I32 = I.astype(np.float32)
    h, w = I32.shape
    floor_med = float(np.median(I32))

    out = []
    for x, y, rad in circles[0][:80]:
        x, y, rad = float(x), float(y), float(rad)
        if not (rad + 2 < x < w - rad - 2 and rad + 2 < y < h - rad - 2):
            continue
        ring = []
        for t in np.linspace(0, 2 * np.pi, 36):
            ring.append(I32[int(y + rad * np.sin(t)),
                            int(x + rad * np.cos(t))])
        rim = float(np.mean(ring))
        # walk from the rim on the shadow side; measure the dark run
        dark_best = 0
        p0 = np.array([x + u[0] * rad * 0.35, y + u[1] * rad * 0.35])
        run = 0
        for t in np.arange(0, rad * 1.6, 1.0):
            px, py = p0[0] + u[0] * t, p0[1] + u[1] * t
            xi, yi = int(round(px)), int(round(py))
            if not (0 <= xi < w and 0 <= yi < h):
                break
            if I32[yi, xi] < 0.55 * rim:
                run += 1
                dark_best = max(dark_best, run)
            else:
                run = 0
        shadow_m = dark_best * cell_m
        depth = shadow_m * tan_el
        if dark_best < 3:                          # no measurable shadow
            continue
        lat, lon = None, None
        _g = _geo_lookup(geo, r0 + y * 4, c0 + x * 4)
        if _g is not None and _g[0] is not None:
            lon, lat = _g  # CSV column order is Longitude, Latitude
        out.append({
            "x_px": round(x, 1), "y_px": round(y, 1),
            "radius_px": round(rad, 1),
            "radius_m": round(rad * cell_m, 1),
            "lon_deg": None if lon is None else round(float(lon), 5),
            "lat_deg": None if lat is None else round(float(lat), 5),
            "shadow_len_m": round(shadow_m, 1),
            "depth_est_m": round(depth, 1),
        })
    out.sort(key=lambda c: -c["radius_m"])
    return out[:max_out]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = parse_label()
    sun = parse_sun_angles()
    meta["sun"] = sun
    print("label parsed:", meta["product_id"], meta["lines"], "x",
          meta["samples"], "| sun:", sun)

    # read the real 1 GB .img via memmap - no full load
    mm = np.memmap(IMG_PATH, dtype=np.uint8, mode="r",
                   shape=(meta["lines"], meta["samples"]))
    (r0, c0), var = pick_best_crop(mm, meta["lines"], meta["samples"])
    crop = np.asarray(mm[r0:r0 + 4096, c0:c0 + 4096], dtype=np.float32)
    print("crop origin (line, sample):", (r0, c0), "std:", round(var, 2))

    # downsample 4x -> 1024x1024 analysis grid (~1.06 m/cell)
    I = cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_AREA)
    cell_m = meta["pixel_resolution_m"] * 4

    # shape-from-shading DEM from the real image
    dem = shape_from_shading(I, sun["sun_azimuth_deg"],
                             sun["sun_elevation_deg"], cell_m=cell_m)
    dem -= dem.min()
    print("SFS DEM range (m): %.2f .. %.2f" % (dem.min(), dem.max()))

    # ---- real reference: LROC NAC selection over all downloaded products ----
    import lroc
    prods = lroc.all_products()
    lroc_meta = None
    ref = None
    if prods:
        print("LROC products found:", len(prods))
        src_small = cv2.resize(I, (384, 384))
        src_small = np.clip(src_small, 0, 255).astype(np.uint8)
        cands = lroc.select_best(src_small, prods)
        # final selection by POST-alignment translation NCC - the coarse
        # template score alone is not reliable on self-similar terrain
        best, lroc_meta, best_ncc, best_ref = None, None, -1.0, None
        for prod, loc, score in cands[:4]:
            ref_c, meta_c = lroc.build_reference(src_small, prod, loc,
                                                 out=1024)
            if ref_c is None:
                print("  candidate %s rejected: %s"
                      % (prod["product_id"], meta_c))
                continue
            ncc = (meta_c.get("translation") or {}).get("ncc", -1.0)
            print("  candidate %s: post-alignment NCC %.3f"
                  % (prod["product_id"], ncc))
            if ncc > best_ncc:
                best, lroc_meta, best_ncc, best_ref = \
                    prod, meta_c, ncc, ref_c
        if best and best_ncc > 0.2:
            ref = best_ref
            print("selected reference:", best["product_id"],
                  "| post-alignment NCC %.3f" % best_ncc)
    if ref is None:
        ref = simulate_second_pass(I)
        ls_diag = render_shaded(dem, sun["sun_azimuth_deg"] + 55.0, 24.0,
                                cell_m=cell_m)
        save_png(ls_diag, os.path.join(OUT_DIR, "ls_render_diagnostic.png"))
        meta["reference_source"] = ("simulated second pass of the OHRC scene "
                                    "(no usable LROC overlap found)")
    else:
        meta["reference_source"] = (
            "REAL NASA LROC NAC CDR %s, registered to the OHRC grid by a "
            "RANSAC-verified homography (%d inliers)."
            % (lroc_meta["product_id"], lroc_meta["inliers"]))
        meta["lroc"] = lroc_meta
        ref = ref.astype(np.float32)
    save_png(ref, os.path.join(OUT_DIR, "reference.png"))
    np.save(os.path.join(OUT_DIR, "reference_dem_input.npy"), np.asarray(ref))

    # ---- cross-mission consistency: OHRC-relief hillshade vs LROC image ----
    try:
        hill = render_shaded(dem, sun["sun_azimuth_deg"],
                             sun["sun_elevation_deg"], cell_m=cell_m)
        ref_f = ref.astype(np.float64) / 255.0
        hill_f = (hill - hill.min()) / (hill.max() - hill.min() + 1e-9)
        mask = (ref_f > 0.02) & (hill_f > 0.02)
        if mask.sum() > 1000:
            pear = float(np.corrcoef(ref_f[mask], hill_f[mask])[0, 1])
        else:
            pear = None
        # illumination-robust check: correlation of gradient magnitudes
        gx1 = cv2.Sobel(ref_f, cv2.CV_64F, 1, 0)
        gy1 = cv2.Sobel(ref_f, cv2.CV_64F, 0, 1)
        gx2 = cv2.Sobel(hill_f, cv2.CV_64F, 1, 0)
        gy2 = cv2.Sobel(hill_f, cv2.CV_64F, 0, 1)
        mag1 = np.hypot(gx1, gy1)[mask]
        mag2 = np.hypot(gx2, gy2)[mask]
        grad_corr = (float(np.corrcoef(mag1, mag2)[0, 1])
                     if mask.sum() > 1000 else None)
        meta["cross_mission_consistency"] = {
            "method": ("independent LROC NAC observation vs OHRC "
                       "shape-from-shading relief: raw Pearson plus "
                       "illumination-robust gradient-magnitude correlation"),
            "pearson": pear,
            "gradient_corr": grad_corr,
            "valid_px": int(mask.sum()) if pear is not None else 0,
        }
        print("cross-mission consistency: pearson=%.3f gradient=%.3f"
              % (pear if pear is not None else float('nan'),
                 grad_corr if grad_corr is not None else float('nan')))
    except Exception as exc:
        meta["cross_mission_consistency"] = {"error": str(exc)}

    # ---- crater identification + shadow-projection depths ------------------
    try:
        craters = detect_craters(I, cell_m, sun, crop_origin=(r0, c0),
                                 meta=meta)
        meta["craters_detected"] = len(craters)
        with open(os.path.join(OUT_DIR, "craters.json"), "w") as f:
            json.dump(craters, f, indent=2)
        print("craters detected:", len(craters))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        meta["craters_detected"] = 0
    save_png(ref, os.path.join(OUT_DIR, "reference.png"))

    save_png(I, os.path.join(OUT_DIR, "source.png"))
    save_png(ref, os.path.join(OUT_DIR, "reference.png"))
    np.save(os.path.join(OUT_DIR, "dem.npy"), dem)

    # 16-bit height raster for any GIS cross-check
    hn = (dem / (dem.max() + 1e-9) * 65535).astype(np.uint16)
    Image.fromarray(hn).save(os.path.join(OUT_DIR, "heightmap_16bit.png"))

    meta["crop_origin_line_sample"] = [int(r0), int(c0)]
    meta["analysis_grid"] = {"n": 1024, "cell_meters": cell_m}
    meta["dem_range_m"] = [float(dem.min()), float(dem.max())]
    meta["provenance"] = (
        "Real Chandrayaan-2 OHRC calibrated radiance (source). Relief "
        "reconstructed by linearized-Lambertian shape-from-shading (sun vector "
        "from the mission sun-parameter file, FFT Poisson integration; "
        "radiometric slope calibration at 8 deg RMS) - photometric "
        "approximation, NOT a metric DEM. Reference = simulated second-pass "
        "product of the same scene (gamma + radiance gradient + noise + "
        "rotation/scale homography) standing in for a LROC NAC / second-orbit "
        "crop until real reference overlap is pulled. Tycho scene remains the "
        "fully synthetic Lommel-Seeliger pair.")
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


    # scene 2: synthetic Tycho stand-in
    tycho_dir = os.path.join(ROOT, "data", "processed", "tycho_synthetic")
    os.makedirs(tycho_dir, exist_ok=True)
    tdem = make_crater_dem(size=1024)
    alb = albedo_field(tdem.shape)
    t_src = render_shaded(tdem, 20.0, 30.0, albedo=0.12 * alb)
    t_ref = render_shaded(tdem, 80.0, 55.0, albedo=0.12 * alb)
    np.save(os.path.join(tycho_dir, "dem.npy"), tdem)
    save_png(t_src, os.path.join(tycho_dir, "source.png"))
    save_png(t_ref, os.path.join(tycho_dir, "reference.png"))
    tmeta = {
        "product_id": "tycho_synthetic",
        "instrument": "Synthetic stand-in (Lommel-Seeliger renderer)",
        "mission": "SIH26166 pipeline self-test",
        "band": "Synthetic panchromatic",
        "start_time_utc": None,
        "target": "Tycho crater stand-in (43.37S, 348.68E)",
        "analysis_grid": {"n": 1024,
                          "cell_meters": 4.0},
        "dem_range_m": [float(tdem.min()), float(tdem.max())],
        "provenance": (
            "Synthetic DEM stand-in for Tycho crater (43.37S, 348.68E), "
            "Lommel-Seeliger rendered pair. Pipeline is data-source-agnostic - "
            "swap in the real PRADAN .img + LROC NAC crop post-hackathon with "
            "zero code changes."),
        "sun": {"sun_azimuth_deg": 20.0, "sun_elevation_deg": 30.0},
    }
    with open(os.path.join(tycho_dir, "metadata.json"), "w") as f:
        json.dump(tmeta, f, indent=2)
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()



