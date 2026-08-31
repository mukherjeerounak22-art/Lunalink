"""Local ONNX descriptor training - the Kaggle notebook (Tomorrow Plan,
Part 2) runnable on CPU. Produces backend/models/descriptor.onnx and
verifies it end-to-end with onnxruntime, so the artifact is guaranteed.

Reduced config vs Kaggle (CPU-friendly): 500 triplets x 4 epochs. Re-run the
Kaggle notebook later for the full T4x2 model and drop it in the same path.
"""
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "backend", "models")
OUT_PATH = os.path.join(OUT_DIR, "descriptor.onnx")
os.makedirs(OUT_DIR, exist_ok=True)

N_PAIRS = int(os.environ.get("N_PAIRS", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
N_TRIALS = int(os.environ.get("N_TRIALS", "100"))


def make_crater_dem(size=512, rim_radius=0.42, depth=1900.0, rim_height=350.0,
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


def albedo_field(shape, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float32)
    for res, amp in [(16, 0.10), (48, 0.09), (128, 0.06)]:
        f = rng.standard_normal((res, res)).astype(np.float32)
        f = np.array(Image.fromarray(f).resize((shape[1], shape[0]),
                                               Image.BICUBIC))
        f = (f - f.mean()) / (f.std() + 1e-9)
        out += amp * f
    return 1.0 + np.clip(out, -0.35, 0.35)


def render_shaded(patch, az_deg, el_deg, albedo=0.12, cell_m=1.0):
    patch = patch.astype(np.float32)
    gy, gx = np.gradient(patch / cell_m)
    normal = np.dstack([-gx, -gy, np.ones_like(patch)]).astype(np.float32)
    normal /= (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8)
    az, el = np.radians(az_deg), np.radians(el_deg)
    sun = np.array([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az),
                    np.sin(el)], dtype=np.float32)
    cos_i = np.clip(normal @ sun, 0, 1)
    a = np.broadcast_to(np.asarray(albedo, dtype=np.float32),
                        patch.shape).astype(np.float32)
    b = a * cos_i / (cos_i + 1e-8)
    return b / (b.max() + 1e-8)


def make_triplet(dems, albs, size, rng):
    di = rng.integers(0, len(dems))
    h, w = dems[di].shape
    r0 = rng.integers(0, h - size)
    c0 = rng.integers(0, w - size)
    patch = dems[di][r0:r0 + size, c0:c0 + size]
    alb = albs[di][r0:r0 + size, c0:c0 + size]
    az = rng.uniform(0, 360)
    anchor = render_shaded(patch, az, rng.uniform(15, 40), 0.12 * alb)
    positive = render_shaded(patch, (az + rng.uniform(20, 60)) % 360,
                             rng.uniform(30, 60), 0.12 * alb)
    dj = rng.integers(0, len(dems))
    r1 = rng.integers(0, 512 - size)
    c1 = rng.integers(0, 512 - size)
    negative = render_shaded(dems[dj][r1:r1 + size, c1:c1 + size],
                             rng.uniform(0, 360), rng.uniform(15, 70),
                             0.12 * albs[dj][r1:r1 + size, c1:c1 + size])
    t = lambda z: torch.from_numpy(z[None, :, :].astype(np.float32))
    return t(anchor), t(positive), t(negative)


class PatchEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        return F.normalize(self.fc(self.net(x).flatten(1)), dim=1)


def triplet_loss(a, p, n, margin=0.2):
    d_pos = (a - p).pow(2).sum(1).sqrt()
    d_neg = (a - n).pow(2).sum(1).sqrt()
    return torch.clamp(d_pos - d_neg + margin, min=0).mean()


def main():
    t0 = time.time()
    dems = [make_crater_dem(size=512, seed=s) for s in (7, 21, 99)]
    albs = [albedo_field(d.shape, seed=s) for d, s in zip(dems, (11, 12, 13))]
    rng = np.random.default_rng(0)
    model = PatchEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("training: %d triplets x %d epochs (CPU)" % (N_PAIRS, EPOCHS))

    for epoch in range(EPOCHS):
        model.train()
        total, count = 0.0, 0
        for _ in range(max(1, N_PAIRS // 8)):
            trips = [make_triplet(dems, albs, 128, rng) for _ in range(8)]
            a = torch.stack([t[0] for t in trips])
            p = torch.stack([t[1] for t in trips])
            n = torch.stack([t[2] for t in trips])
            opt.zero_grad()
            loss = triplet_loss(model(a), model(p), model(n))
            loss.backward()
            opt.step()
            total += loss.item() * 8
            count += 8
        print("epoch %d/%d  loss %.4f  (%.0fs)"
              % (epoch + 1, EPOCHS, total / count, time.time() - t0))

    # validation: triplet ranking accuracy d(a,p) < d(a,n)
    model.eval()
    correct = 0
    vrng = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(N_TRIALS):
            a, p, n = make_triplet(dems, albs, 128, vrng)
            ea, ep, en = model(a[None]), model(p[None]), model(n[None])
            correct += ((ea - ep).pow(2).sum() <
                        (ea - en).pow(2).sum()).item()
    acc = correct / N_TRIALS
    print("triplet ranking accuracy: %.3f (target > 0.5, ideally > 0.9)" % acc)

    # ONNX export
    model.eval()
    dummy = torch.randn(1, 1, 128, 128)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, dummy, OUT_PATH,
                          input_names=["patch"], output_names=["embedding"],
                          opset_version=17, dynamo=False,
                          dynamic_axes={"patch": {0: "batch"},
                                        "embedding": {0: "batch"}})
    print("exported:", OUT_PATH, "(%.2f MB)"
          % (os.path.getsize(OUT_PATH) / 1e6))

    # end-to-end verification with onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"patch": np.random.rand(1, 1, 128, 128)
                          .astype(np.float32)})[0]
    assert out.shape == (1, 128), out.shape
    norm = float(np.linalg.norm(out))
    assert abs(norm - 1.0) < 0.01, norm
    print("onnxruntime verification OK: embedding shape %s, L2 norm %.4f"
          % (out.shape, norm))
    print("DONE in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
