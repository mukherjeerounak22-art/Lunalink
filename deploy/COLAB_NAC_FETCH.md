# Colab → Kaggle: fetch the polar NAC strips WITHOUT using laptop storage

Runs entirely on Google Colab (fast pipe, no laptop disk). Produces a small
public Kaggle dataset `lro-nac-polar` containing ONLY the strips that
actually cover the Option-B region — which is what `KAGGLE_LRO_DATASET`
must point at (the runtime fetch pulls the whole dataset, so it has to stay
small — never upload all 115 URLs).

**Before starting:** download the small ranker reference —
`data/reference/tmc/ch2_tmc_ndn_20231109T2148028796_d_dtm_d18/browse.png`
(~292 KB) — from the repo; Cell 2 asks for it. Also grab your
`kaggle.json` (kaggle.com → Settings → API → Create New Token).

---

## Cell 1 — download the candidate strips (spread across years)

```python
import os, subprocess

URLS = [
 "http://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1067A/DATA/ESM6/2026104/NAC/M1530835345LC.IMG",
 "http://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1066C/DATA/ESM6/2026050/NAC/M1526158258LC.IMG",
 "http://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1064B/DATA/ESM5/2025199/NAC/M1507447398LC.IMG",
 "http://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1061A/DATA/ESM5/2024267/NAC/M1481695759LC.IMG",
 "http://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1050A/DATA/ESM4/2021358/NAC/M1394959095LC.IMG",
 "http://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1041B/DATA/ESM4/2019300/NAC/M1326856794LC.IMG",
 "http://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1025/DATA/ESM2/2015319/NAC/M1202234505LC.IMG",
 "http://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1001/DATA/MAP/2009288/NAC/M110208956LC.IMG",
]
os.makedirs("/content/strips", exist_ok=True)
for u in URLS:
    n = os.path.join("/content/strips", u.split("/")[-1])
    if not os.path.exists(n):
        subprocess.run(["wget", "-q", "-O", n, u], check=True)
    print(n, "%.0f MB" % (os.path.getsize(n) / 1e6))
```

## Cell 2 — upload the small ranker reference (browse.png, ~292 KB)

```python
from google.colab import files
up = files.upload()                      # select browse.png
BROWSE = "/content/browse.png"
```

## Cell 3 — rank the strips: which ones cover the Option-B region?

Same empirical logic as the backend: coarse-match each strip against the
TMC-2 DTM browse of the target region. (The equatorial strips scored only
≤0.18 against this browse — a covering strip will score clearly higher.)

```python
import cv2, numpy as np, glob, os

browse = cv2.imread("/content/browse.png", cv2.IMREAD_GRAYSCALE)
src = cv2.resize(browse, (1024, 1024), interpolation=cv2.INTER_AREA)

def preview(path, factor=8):
    mm = np.memmap(path, dtype="<i2", mode="r", offset=5064,
                   shape=(52224, 5064))
    sub = np.asarray(mm[::factor, ::factor], dtype=np.float32)
    lo, hi = np.nanpercentile(sub, 2), np.nanpercentile(sub, 98)
    v = np.clip((sub - lo) / (hi - lo + 1e-9) * 255, 0, 255)
    return np.nan_to_num(v).astype(np.uint8)

results = []
for p in sorted(glob.glob("/content/strips/*.IMG")):
    prev = preview(p)
    size = min(542, prev.shape[0], prev.shape[1])       # OHRC patch @ 4 m/cell
    tpl = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(prev, tpl, cv2.TM_CCOEFF_NORMED)
    _, sc, _, loc = cv2.minMaxLoc(res)
    results.append((float(sc), os.path.basename(p), loc))
    print("%-18s NCC %.3f at %s" % (os.path.basename(p), sc, loc))
results.sort(reverse=True)
print("\nRANKING (covers the region if clearly above ~0.20):")
for sc, n, loc in results:
    print("  %.3f  %s" % (sc, n))
```

## Cell 4 — publish ONLY the winners to Kaggle

```python
# keep the top strips (raise/lower the cut after seeing Cell 3's spread)
KEEP = [n for sc, n, loc in results[:3]]
os.makedirs("/content/keep", exist_ok=True)
import shutil
for n in KEEP:
    shutil.copy("/content/strips/" + n, "/content/keep/" + n)

from google.colab import files
files.upload()                            # select kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip -q install kaggle

meta = '''{
  "title": "lro-nac-polar",
  "id": "YOUR_KAGGLE_USERNAME/lro-nac-polar",
  "licenses": [{"name": "other"}]
}'''
open("/content/keep/dataset-metadata.json", "w").write(meta)
!kaggle datasets create -p /content/keep --dir-mode zip
```

## Cell 5 — verify it round-trips (exactly what the backend does)

```python
!pip -q install kagglehub
import kagglehub
p = kagglehub.dataset_download("YOUR_KAGGLE_USERNAME/lro-nac-polar")
print(p, os.listdir(p))
```

---

After this: set the Render/GitHub secret `KAGGLE_LRO_DATASET = <you>/lro-nac-polar`.
The backend's `lroc.ensure_library()` pulls exactly this dataset on demand.

**Still needed from PRADAN (browser download → upload straight to Kaggle via
the web UI, nothing kept locally):** the polar **OHRC ortho product** over
lat −68.39…−67.39, lon 209.9–210.9 — it is the scene anchor and the last
missing instrument.
