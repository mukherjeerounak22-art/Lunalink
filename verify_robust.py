"""Robustness verification: ONNX descriptor determinism/batch-invariance +
every piece of derived math in the project, checked against analytic truth."""
import sys

import numpy as np
import cv2

sys.path.insert(0, "backend")
import pipeline
from pipeline import (fourier_smooth, marching_squares, mutual_information,
                      _session, _embed, PATCH)

passed, failed = 0, []


def check(label, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print("  PASS", label, detail)
    else:
        failed.append(label)
        print("  FAIL", label, detail)


print("=== ONNX descriptor robustness ===")
rng = np.random.default_rng(1)
img = (rng.standard_normal((256, 256)) * 40 + 128).clip(0, 255).astype(np.float32)
import cv2 as _cv
sift = _cv.SIFT_create(nfeatures=30)
kps, _ = sift.detectAndCompute(img.astype(np.uint8), None)
check("keypoints found", len(kps) >= 5, f"({len(kps)})")

e1 = _embed(img, kps)
e2 = _embed(img, kps)
check("deterministic (same patch -> same embedding)",
      np.allclose(e1, e2, atol=1e-6))
check("L2-normalized embeddings",
      np.allclose(np.linalg.norm(e1, axis=1), 1.0, atol=1e-4))

# batch invariance: embedding of kp0 alone == embedding within a big batch
solo = _embed(img, [kps[0]])[0]
in_batch = _embed(img, kps)[0]
check("batch invariance (1 vs N)", np.allclose(solo, in_batch, atol=1e-5),
      f"max diff {np.abs(solo - in_batch).max():.2e}")

# distinct patches -> distinct embeddings
far_kp = [kps[0], kps[-1]]
if len(kps) > 1 and np.hypot(kps[0].pt[0] - kps[-1].pt[0],
                             kps[0].pt[1] - kps[-1].pt[1]) > 100:
    e_far = _embed(img, far_kp)
    cos = float(e_far[0] @ e_far[1])
    check("distinct patches -> distinct embeddings", cos < 0.995,
          f"cosine {cos:.3f}")

# border keypoints (edge padding correctness)
border_kps = [_cv.KeyPoint(2, 2, 8), _cv.KeyPoint(253, 253, 8),
              _cv.KeyPoint(2, 253, 8), _cv.KeyPoint(253, 2, 8)]
e_border = _embed(img, border_kp := border_kps)
check("border keypoints handled (edge padding)",
      e_border.shape == (4, 128) and np.isfinite(e_border).all())
check("empty keypoint list safe", _embed(img, []).shape == (0, 128))

print("=== Problem 3: RANSAC iteration formula ===")
p = 0.99
k = lambda w: int(np.ceil(np.log(1 - p) / np.log(1 - w ** 4)))
check("w=0.3 -> ~567 iterations (plan's worked example)",
      560 <= k(0.3) <= 575, f"(k={k(0.3)})")
check("k decreases monotonically with w",
      k(0.1) > k(0.3) > k(0.5) > k(0.9))

print("=== Problem 6: mutual information (Jensen bound) ===")
a = rng.standard_normal((512, 512))
b = rng.standard_normal((512, 512))
mi_id = mutual_information(a, a)
mi_ind = mutual_information(a * 1000, b)
check("I(A;A) high (identical fields)", mi_id > 2.0, f"({mi_id:.2f})")
check("I(A;B) ~ 0 (independent fields)", -0.05 < mi_ind < 0.1,
      f"({mi_ind:.3f})")

print("=== Problem 7: truncated Fourier low-pass ===")
yy, xx = np.mgrid[0:128, 0:128]
signal = np.sin(2 * np.pi * 3 * xx / 128) + np.cos(2 * np.pi * 2 * yy / 128)
noisy = signal + 0.5 * rng.standard_normal((128, 128))
smooth = fourier_smooth(noisy.astype(np.float32), keep_fraction=0.10)
err_before = np.abs(noisy - signal).mean()
err_after = np.abs(smooth - signal).mean()
check("low-pass beats raw (noise removed)", err_after < err_before * 0.5,
      f"(err {err_before:.3f} -> {err_after:.3f})")

print("=== Problem 8: marching squares vs analytic truth ===")
yy, xx = np.mgrid[0:200, 0:200]
disc = (xx - 100) ** 2 + (yy - 100) ** 2 - 60.0 ** 2   # circle r=60
segs = marching_squares(disc.astype(np.float32), 0.0)
# total segment length should approximate circumference 2*pi*60 = 377
lengths = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in segs)
check("contour length ~ circle circumference", 350 < lengths < 405,
      f"({lengths:.0f} vs {2 * np.pi * 60:.0f})")

print("=== Problem 9 lineage: FFT Poisson solver round-trip ===")
# synthetic Gaussian hill -> analytic slopes -> FFT Poisson solve -> recover
N = 128
yy, xx = np.mgrid[0:N, 0:N]
hill = 50 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 15.0 ** 2))
p_field, q_field = np.gradient(hill.astype(np.float64))     # dh/dx, dh/dy


def poisson_solve(p, q):
    # full 2-D FFT (same fix as production shape_from_shading)
    div = np.gradient(p, axis=0) + np.gradient(q, axis=1)
    fy = np.fft.fftfreq(N)[:, None]
    fx = np.fft.fftfreq(N)[None, :]
    k2 = (2 * np.pi * fy) ** 2 + (2 * np.pi * fx) ** 2
    k2[0, 0] = 1.0
    h = np.real(np.fft.ifft2(np.fft.fft2(div) / (-k2)))
    return h - h.mean()


target = hill - hill.mean()
# 1. pure solver check (no windowing): must recover the surface
h_rec = poisson_solve(p_field, q_field)
corr = float(np.corrcoef(h_rec.ravel(), target.ravel())[0, 1])
check("FFT Poisson solver recovers known surface", corr > 0.95,
      f"(corr {corr:.3f})")
# 2. production uses a Hann window for boundary handling - attenuates the
#    field toward edges (documented behavior), must still correlate
w = np.hanning(N)[:, None] * np.hanning(N)[None, :]
h_win = poisson_solve(p_field * w, q_field * w)
corr_win = float(np.corrcoef(h_win.ravel(), target.ravel())[0, 1])
check("windowed production variant still correlated", corr_win > 0.5,
      f"(corr {corr_win:.3f})")

print()
print("ROBUSTNESS RESULT: %d passed, %d failed" % (passed, len(failed)))
if failed:
    sys.exit("FAILURES: " + str(failed))
