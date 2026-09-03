# Colab: download EVERYTHING → publish to Kaggle (laptop stays empty)

One session, one notebook. It fetches **all 115 NAC strips** from the LROC
mirror, **all OHRC products from your PRADAN order**, ranks the NAC strips
so only region-covering ones get published, and pushes everything to your
Kaggle account as public datasets. The laptop is never involved.

**You need:** your `kaggle.json` (kaggle.com → Settings → API → Create New
Token), the small `browse.png` of the polar DTM (from
`data/reference/tmc/ch2_tmc_ndn_20231109T2148028796_d_dtm_d18/`), and — for
Cell 4 — a **freshly generated PRADAN download script** (log into PRADAN,
re-select the OHRC products, "Download the generated script", keep the
browser session open while Cell 4 runs).

⚠️ Colab limits: ~12 h session, ~100 GB temp disk (55 GB of NAC strips
fits). Don't let the tab idle — Colab disconnects.

---

## Cell 1 — setup + Kaggle credentials

```python
from google.colab import files
!pip -q install kaggle kagglehub

up = files.upload()                       # select kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
print("kaggle ready")
```

## Cell 2 — download ALL 115 NAC strips (~55 GB, 30–60 min)

```python
import os, subprocess

BASE = "http://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0"
PATHS = """
LROLRC_1067A/DATA/ESM6/2026104/NAC/M1530835345LC.IMG
LROLRC_1067A/DATA/ESM6/2026104/NAC/M1530828332LC.IMG
LROLRC_1067A/DATA/ESM6/2026077/NAC/M1528500333LC.IMG
LROLRC_1066C/DATA/ESM6/2026050/NAC/M1526158258LC.IMG
LROLRC_1065A/DATA/ESM5/2025268/NAC/M1513481707LC.IMG
LROLRC_1064B/DATA/ESM5/2025199/NAC/M1507447398LC.IMG
LROLRC_1064B/DATA/ESM5/2025199/NAC/M1507440401LC.IMG
LROLRC_1061C/DATA/ESM5/2024321/NAC/M1486374074LC.IMG
LROLRC_1061A/DATA/ESM5/2024282/NAC/M1483041531LC.IMG
LROLRC_1061A/DATA/ESM5/2024267/NAC/M1481695759LC.IMG
LROLRC_1061A/DATA/ESM5/2024266/NAC/M1481688745LC.IMG
LROLRC_1061A/DATA/ESM5/2024266/NAC/M1481681728LC.IMG
LROLRC_1059B/DATA/ESM4/2024119/NAC/M1468987187LC.IMG
LROLRC_1059A/DATA/ESM4/2024104/NAC/M1467634552LC.IMG
LROLRC_1058B/DATA/ESM4/2024038/NAC/M1461946398LC.IMG
LROLRC_1057C/DATA/ESM4/2023349/NAC/M1457260005LC.IMG
LROLRC_1056C/DATA/ESM4/2023252/NAC/M1448895321LC.IMG
LROLRC_1056C/DATA/ESM4/2023252/NAC/M1448888299LC.IMG
LROLRC_1056B/DATA/ESM4/2023225/NAC/M1446542838LC.IMG
LROLRC_1055C/DATA/ESM4/2023143/NAC/M1439507705LC.IMG
LROLRC_1054C/DATA/ESM4/2023050/NAC/M1431465061LC.IMG
LROLRC_1054A/DATA/ESM4/2022361/NAC/M1426775561LC.IMG
LROLRC_1054A/DATA/ESM4/2022361/NAC/M1426768517LC.IMG
LROLRC_1053A/DATA/ESM4/2022264/NAC/M1418414438LC.IMG
LROLRC_1052A/DATA/ESM4/2022183/NAC/M1411372754LC.IMG
LROLRC_1051B/DATA/ESM4/2022116/NAC/M1405656639LC.IMG
LROLRC_1051A/DATA/ESM4/2022089/NAC/M1403313181LC.IMG
LROLRC_1050B/DATA/ESM4/2022035/NAC/M1398614027LC.IMG
LROLRC_1050A/DATA/ESM4/2021358/NAC/M1394959095LC.IMG
LROLRC_1050A/DATA/ESM4/2021358/NAC/M1394952091LC.IMG
LROLRC_1049B/DATA/ESM4/2021303/NAC/M1390260350LC.IMG
LROLRC_1049A/DATA/ESM4/2021276/NAC/M1387907574LC.IMG
LROLRC_1048C/DATA/ESM4/2021249/NAC/M1385569219LC.IMG
LROLRC_1048B/DATA/ESM4/2021222/NAC/M1383217896LC.IMG
LROLRC_1048A/DATA/ESM4/2021183/NAC/M1379838943LC.IMG
LROLRC_1047B/DATA/ESM4/2021128/NAC/M1375140212LC.IMG
LROLRC_1046C/DATA/ESM4/2021074/NAC/M1370440757LC.IMG
LROLRC_1046C/DATA/ESM4/2021047/NAC/M1368092731LC.IMG
LROLRC_1046B/DATA/ESM4/2021032/NAC/M1366785548LC.IMG
LROLRC_1045C/DATA/ESM4/2020343/NAC/M1362092945LC.IMG
LROLRC_1045A/DATA/ESM4/2020262/NAC/M1355046602LC.IMG
LROLRC_1044C/DATA/ESM4/2020234/NAC/M1352690370LC.IMG
LROLRC_1044A/DATA/ESM4/2020168/NAC/M1346953031LC.IMG
LROLRC_1043A/DATA/ESM4/2020086/NAC/M1339904129LC.IMG
LROLRC_1042C/DATA/ESM4/2020059/NAC/M1337554808LC.IMG
LROLRC_1042C/DATA/ESM4/2020059/NAC/M1337547788LC.IMG
LROLRC_1042B/DATA/ESM4/2020017/NAC/M1333911517LC.IMG
LROLRC_1042A/DATA/ESM4/2019355/NAC/M1331560634LC.IMG
LROLRC_1042A/DATA/ESM4/2019355/NAC/M1331553562LC.IMG
LROLRC_1041C/DATA/ESM4/2019328/NAC/M1329208234LC.IMG
LROLRC_1041C/DATA/ESM4/2019328/NAC/M1329201180LC.IMG
LROLRC_1041B/DATA/ESM4/2019300/NAC/M1326856794LC.IMG
LROLRC_1040C/DATA/ESM3/2019234/NAC/M1321108670LC.IMG
LROLRC_1039A/DATA/ESM3/2019098/NAC/M1309351889LC.IMG
LROLRC_1038C/DATA/ESM3/2019071/NAC/M1306998683LC.IMG
LROLRC_1038B/DATA/ESM3/2019029/NAC/M1303366359LC.IMG
LROLRC_1038A/DATA/ESM3/2019001/NAC/M1301011052LC.IMG
LROLRC_1037B/DATA/ESM3/2018312/NAC/M1296309147LC.IMG
LROLRC_1037A/DATA/ESM3/2018285/NAC/M1293963112LC.IMG
LROLRC_1036C/DATA/ESM3/2018245/NAC/M1290542732LC.IMG
LROLRC_1036B/DATA/ESM3/2018218/NAC/M1288191192LC.IMG
LROLRC_1035/DATA/ESM3/2018136/NAC/M1281137806LC.IMG
LROLRC_1034/DATA/ESM3/2018067/NAC/M1275156611LC.IMG
LROLRC_1034/DATA/ESM3/2018067/NAC/M1275149618LC.IMG
LROLRC_1034/DATA/ESM3/2017350/NAC/M1268098488LC.IMG
LROLRC_1033/DATA/ESM3/2017323/NAC/M1265742550LC.IMG
LROLRC_1033/DATA/ESM3/2017296/NAC/M1263400700LC.IMG
LROLRC_1033/DATA/ESM3/2017296/NAC/M1263393624LC.IMG
LROLRC_1032/DATA/ESM3/2017202/NAC/M1255269724LC.IMG
LROLRC_1031/DATA/ESM3/2017120/NAC/M1248215306LC.IMG
LROLRC_1030/DATA/ESM3/2017024/NAC/M1239889052LC.IMG
LROLRC_1030/DATA/ESM3/2016363/NAC/M1237533054LC.IMG
LROLRC_1030/DATA/ESM3/2016363/NAC/M1237526033LC.IMG
LROLRC_1029/DATA/ESM3/2016308/NAC/M1232821098LC.IMG
LROLRC_1028/DATA/ESM2/2016187/NAC/M1222326902LC.IMG
LROLRC_1027/DATA/ESM2/2016159/NAC/M1219977753LC.IMG
LROLRC_1027/DATA/ESM2/2016159/NAC/M1219970723LC.IMG
LROLRC_1027/DATA/ESM2/2016132/NAC/M1217621583LC.IMG
LROLRC_1027/DATA/ESM2/2016118/NAC/M1216356418LC.IMG
LROLRC_1027/DATA/ESM2/2016090/NAC/M1214000331LC.IMG
LROLRC_1025/DATA/ESM2/2015346/NAC/M1204590436LC.IMG
LROLRC_1025/DATA/ESM2/2015319/NAC/M1202234505LC.IMG
LROLRC_1023/DATA/ESM2/2015143/NAC/M1187013686LC.IMG
LROLRC_1023/DATA/ESM2/2015101/NAC/M1183416764LC.IMG
LROLRC_1023/DATA/ESM2/2015101/NAC/M1183409613LC.IMG
LROLRC_1022/DATA/ESM2/2014357/NAC/M1173995258LC.IMG
LROLRC_1021/DATA/ESM2/2014263/NAC/M1165828686LC.IMG
LROLRC_1019/DATA/ESM/2014139/NAC/M1155167580LC.IMG
LROLRC_1019/DATA/ESM/2014112/NAC/M1152811365LC.IMG
LROLRC_1019/DATA/ESM/2014085/NAC/M1150454976LC.IMG
LROLRC_1018/DATA/ESM/2014058/NAC/M1148098944LC.IMG
LROLRC_1016/DATA/ESM/2013191/NAC/M1128132571LC.IMG
LROLRC_1015/DATA/ESM/2013150/NAC/M1124549266LC.IMG
LROLRC_1014/DATA/ESM/2013068/NAC/M1117479532LC.IMG
LROLRC_1013/DATA/ESM/2012312/NAC/M1106918895LC.IMG
LROLRC_1012/DATA/SCI/2012189/NAC/M1096274701LC.IMG
LROLRC_1011/DATA/SCI/2012161/NAC/M193916828LC.IMG
LROLRC_1011/DATA/SCI/2012107/NAC/M189206507LC.IMG
LROLRC_1010/DATA/SCI/2012025/NAC/M182136962LC.IMG
LROLRC_1008/DATA/SCI/2011240/NAC/M169189601LC.IMG
LROLRC_1008/DATA/SCI/2011172/NAC/M163286664LC.IMG
LROLRC_1008/DATA/SCI/2011172/NAC/M163279879LC.IMG
LROLRC_1007/DATA/SCI/2011145/NAC/M160925346LC.IMG
LROLRC_1007/DATA/SCI/2011117/NAC/M158570331LC.IMG
LROLRC_1006/DATA/SCI/2011063/NAC/M153848055LC.IMG
LROLRC_1005/DATA/SCI/2010305/NAC/M143243848LC.IMG
LROLRC_1004/DATA/MAP/2010250/NAC/M138521664LC.IMG
LROLRC_1003/DATA/MAP/2010128/NAC/M127917242LC.IMG
LROLRC_1002/DATA/MAP/2010073/NAC/M123200403LC.IMG
LROLRC_1002/DATA/MAP/2010018/NAC/M118477228LC.IMG
LROLRC_1001/DATA/MAP/2009288/NAC/M110208956LC.IMG
LROLRC_1001/DATA/COM/2009233/NAC/M105487356LC.IMG
""".split()

os.makedirs("/content/strips", exist_ok=True)
done = 0
for rel in PATHS:
    n = os.path.join("/content/strips", rel.split("/")[-1])
    url = f"{BASE}/{rel}"
    if os.path.exists(n) and os.path.getsize(n) > 400e6:
        done += 1; continue
    subprocess.run(["wget", "-q", "-c", "-O", n, url], check=False)
    ok = os.path.exists(n) and os.path.getsize(n) > 400e6
    done += ok
    print(("OK  " if ok else "FAIL"), rel.split("/")[-1], f"({done}/{len(PATHS)})")
print(f"\n{done}/{len(PATHS)} strips on disk")
```

---

## Cell 3 — upload the small ranker + rank ALL strips

```python
from google.colab import files
up = files.upload()                       # select browse.png (~292 KB)
BROWSE = "/content/browse.png"
```

## Cell 4 — rank all strips (decides what gets published)

```python
import cv2, numpy as np, glob, os

src = cv2.resize(cv2.imread(BROWSE, cv2.IMREAD_GRAYSCALE),
                 (1024, 1024), interpolation=cv2.INTER_AREA)

def preview(path, factor=8):
    mm = np.memmap(path, dtype="<i2", mode="r", offset=5064,
                   shape=(52224, 5064))
    sub = np.asarray(mm[::factor, ::factor], dtype=np.float32)
    lo, hi = np.nanpercentile(sub, 2), np.nanpercentile(sub, 98)
    return np.nan_to_num(np.clip((sub - lo)/(hi - lo + 1e-9)*255, 0, 255)
                         ).astype(np.uint8)

results = []
for p in sorted(glob.glob("/content/strips/*.IMG")):
    try:
        prev = preview(p)
        size = min(542, prev.shape[0], prev.shape[1])
        tpl = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
        _, sc, _, loc = cv2.matchTemplate(prev, tpl, cv2.TM_CCOEFF_NORMED)
        results.append((float(sc), os.path.basename(p)))
        print("%-18s NCC %.3f" % (os.path.basename(p), sc))
    except Exception as e:
        print(os.path.basename(p), "skipped:", str(e)[:80])
results.sort(reverse=True)
print("\nRANKING — strips scoring well above the ~0.20 noise floor cover "
      "the polar region:")
for sc, n in results[:8]:
    print("  %.3f  %s" % (sc, n))
```

---

## Cell 5 — download the OHRC products from your PRADAN order

Log into PRADAN, re-select the OHRC products, click **"Download the
generated script"**, keep the browser open, then:

```python
up = files.upload()                       # select the generated ohrc_*.py
src = open(list(up.keys())[0], encoding="utf-8", errors="replace").read()

import re, requests
from pathlib import Path
prefix  = re.search(r'url_prefix\s*=\s*"([^"]*)"', src).group(1)
cookie  = re.search(r'cookie_string\s*=\s*"([^"]*)"', src).group(1)
paths   = eval(re.search(r'data_file_paths\s*=\s*(\[.*?\])\s*\n', src, re.S).group(1))
headers = {"Cookie": cookie}
session = requests.Session()

base = Path("/content/pradan")
for rel in paths:
    rel = rel.split("?")[0].strip("/")
    out = base / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1e6:
        print("skip (have):", out.name); continue
    r = session.get(prefix + "/" + rel.strip("/"), headers=headers,
                    stream=True, timeout=(30, 600))
    if r.status_code != 200:
        print("FAIL", r.status_code, rel.split("/")[-1]); continue
    with open(out, "wb") as f:
        for chunk in r.iter_content(8 * 1024 * 1024):
            f.write(chunk)
    print("OK %.1f MB  %s" % (out.stat().st_size / 1e6, out.name))
print("OHRC products in:", base)
```

(If a download fails 401/403, the PRADAN session expired — re-select the
products, generate a new script, re-run Cell 5. Finished files are skipped,
so it resumes.)

---

## Cell 6 — publish to Kaggle (two public datasets)

```python
import shutil, glob, json, os, subprocess

def publish(src_dir, slug, title):
    json.dump({"title": title, "id": f"YOUR_KAGGLE_USERNAME/{slug}",
               "licenses": [{"name": "other"}]},
              open(os.path.join(src_dir, "dataset-metadata.json"), "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", src_dir,
                        "--dir-mode", "zip"], capture_output=True, text=True)
    print(slug, "->", (r.stdout or r.stderr)[-200:])

# 1) NAC strips: only the winners (top 4 by NCC — edit after seeing Cell 4)
os.makedirs("/content/pub_lro", exist_ok=True)
for sc, n in results[:4]:
    shutil.copy("/content/strips/" + n, "/content/pub_lro/" + n)
publish("/content/pub_lro", "lro-nac-polar", "LRO NAC polar strips")

# 2) OHRC products from the PRADAN order
os.makedirs("/content/pub_ohrc", exist_ok=True)
for p in glob.glob("/content/pradan/**/*.zip", recursive=True):
    shutil.copy(p, "/content/pub_ohrc/" + os.path.basename(p))
publish("/content/pub_ohrc", "ohrc-polar", "Chandrayaan-2 OHRC polar products")

# 3) ALSO upload via the Kaggle WEB UI (browser drag-drop, nothing kept
#    locally): ch2_tmc_ndn_20231109T2148028796_d_dtm_d18.zip and
#    ch2_iir_nci_20221227T0748212038_d_img_d32.zip -> dataset "tmc-iirs-polar"
```

---

## Cell 7 — verify the round trip (exactly what the backend does)

```python
import kagglehub, os
for slug in ("lro-nac-polar", "ohrc-polar"):
    p = kagglehub.dataset_download(f"YOUR_KAGGLE_USERNAME/{slug}")
    print(slug, "->", os.listdir(p))
```

---

After this: the two TMC/IIRS zips + the Kaggle datasets are permanent, the
laptop holds nothing, and `KAGGLE_LRO_DATASET = <you>/lro-nac-polar` goes
into the Render environment. Then: local verification pass -> hosting.
