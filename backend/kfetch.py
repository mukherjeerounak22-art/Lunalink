"""On-demand fetch of the big TMC-2/IIRS rasters from public Kaggle
datasets - the remote-hosting counterpart of the local zips.

    KAGGLE_TMC_DATASET = <owner>/<slug>   dataset contains <pid>.zip files
    KAGGLE_IIRS_DATASET = <owner>/<slug>  (e.g. ch2_iir_nci_..._d_img_d32.zip)

Public datasets need no Kaggle credentials.  The wanted product zip is
pulled per-file via kagglehub, and only the wanted member (the .tif DTM
or the .qub spectral cube) is streamed out of it.  Every failure is
graceful: the caller keeps its honest no-data error.
"""
import os
import requests

def _auth():
    u = os.environ.get("KAGGLE_USERNAME")
    k = os.environ.get("KAGGLE_KEY")
    if u and k:
        return (u, k)
    # local fallback: kaggle.json in the repo root (gitignored)
    cand = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "kaggle.json")
    try:
        cfg = json.load(open(cand))
        return (cfg["username"], cfg["key"])
    except Exception:                                    # noqa: BLE001
        return None

def download_dataset_file(owner, slug, file_path, out_path,
                          chunk=8 * 1024 * 1024):
    """Stream one file out of a (public) Kaggle dataset via the v1 API.
    Works for web-UI-created datasets where kagglehub's version
    resolution 404s.  Returns True on success."""
    url = (f"https://www.kaggle.com/api/v1/datasets/download/{owner}/"
           f"{slug}?fileName={requests.utils.quote(file_path)}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, auth=_auth(), stream=True,
                      timeout=(30, 600)) as r:
        if r.status_code != 200:
            print("kaggle-fetch: HTTP", r.status_code, file_path)
            return False
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            for c in r.iter_content(chunk):
                f.write(c)
    os.replace(tmp, out_path)
    return True


def download_dataset_zip(owner, slug, out_path, chunk=8 * 1024 * 1024):
    """Download the FULL dataset as one zip via the v1 API."""
    url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}"
    with requests.get(url, auth=_auth(), stream=True, timeout=(30, 1800)) as r:
        if r.status_code != 200:
            print("kaggle-fetch: HTTP", r.status_code, slug)
            return False
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            for c in r.iter_content(chunk):
                f.write(c)
    os.replace(tmp, out_path)
    return True


def _slug(name):
    return (os.environ.get(name) or "").strip()


def _slugs(name):
    """Env var may list ONE dataset ('owner/slug') or SEVERAL separated by
    commas/spaces - the files are scattered across the user's datasets, so
    every candidate is searched until the wanted product is found."""
    raw = _slug(name)
    out = []
    for part in raw.replace(",", " ").split():
        part = part.strip().strip("/")
        if "/" in part and part not in out:
            out.append(part)
    return out


def _extract_member(zpath, member_suffix, out_path):
    with zipfile.ZipFile(zpath) as z:
        member = next((n for n in z.namelist()
                       if os.path.basename(n).lower()
                       == member_suffix.lower()), None)
        if member is None:
            return False
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".part"
        with z.open(member) as src, open(tmp, "wb") as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    os.replace(tmp, out_path)
    return True


def _candidate_paths(pid, date, kind):
    """In-dataset paths seen in the extracted-layout uploads.  `kind` is
    'dtm' (.tif under data/derived) or 'cube' (.qub under data/calibrated)."""
    if kind == "dtm":
        member = pid + ".tif"
        sub = "derived"
    else:
        member = pid + ".qub"
        sub = "calibrated"
    return [
        f"{pid}/data/{sub}/{date}/{member}",
        f"data/{sub}/{date}/{member}",
        member,
    ]


def _date_of(pid):
    ts = pid.split("T")[0] if "T" in pid else pid
    return ts[-8:]


def _list_dataset_files(slug):
    """Full file list of a Kaggle dataset via the authenticated API."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    names, token = [], None
    for _ in range(40):
        res = api.dataset_list_files(slug, page_token=token or None,
                                     page_size=100)
        for f in (getattr(res, "files", None) or []):
            n = getattr(f, "name", None) or getattr(f, "path", None)
            if n:
                names.append(n)
        token = getattr(res, "next_page_token", None)
        if not token:
            break
    return names

def _fetch_member(slug, pid, kind, out_path):
    date = _date_of(pid)
    owner, dslug = slug.split("/", 1)
    member_name = (pid + (".tif" if kind == "dtm" else ".qub")).lower()
    last_err = None
    # resolve the exact in-dataset path via the file listing (any layout)
    cand_paths = _candidate_paths(pid, date, kind)
    try:
        for n in _list_dataset_files(slug):
            if os.path.basename(n).lower() == member_name:
                cand_paths.insert(0, n)
                break
    except Exception as exc:                             # noqa: BLE001
        print("kfetch: listing failed (%s)" % str(exc)[:100])
    for cand in cand_paths:
        tmp = out_path + ".dl"
        try:
            with open(tmp, "rb") as fh:
                magic = fh.read(2)
            if magic != b"PK":
                os.replace(tmp, out_path)      # per-file download IS the member
                return True
            if _extract_member(tmp, os.path.basename(cand), out_path):
                os.remove(tmp)
                return True
            last_err = "member not in file"
        except Exception as exc:                         # noqa: BLE001
            last_err = str(exc)[:140]
    # fallback: full-dataset zip, walk for the member
    try:
        ztmp = out_path + ".ds.zip"
        if download_dataset_zip(owner, dslug, ztmp):
            suffix = ".tif" if kind == "dtm" else ".qub"
            with zipfile.ZipFile(ztmp) as z:
                member = next((n for n in z.namelist()
                               if os.path.basename(n).lower()
                               == (pid + suffix).lower()), None)
                if member:
                    return _extract_member(ztmp, member, out_path)
            os.remove(ztmp)
    except Exception as exc:                             # noqa: BLE001
        last_err = str(exc)[:140]
    if last_err:
        print("kfetch: candidates failed (%s)" % last_err)
    return False


def ensure_dtm(product, out_tif):
    """dtm.tif for a TMC-2 DTM product, fetched on demand.  True on
    success; False (and the caller's honest error) on any failure.
    Every dataset listed in KAGGLE_TMC_DATASET is searched in turn."""
    pid = product.get("product_id") or ""
    for slug in _slugs("KAGGLE_TMC_DATASET"):
        try:
            if _fetch_member(slug, pid, "dtm", out_tif):
                return True
        except Exception as exc:                         # noqa: BLE001
            print("kfetch: DTM fetch failed on %s (%s)"
                  % (slug, str(exc)[:140]))
    return False


def ensure_cube(product, out_qub):
    """cube.qub for an IIRS product, fetched on demand.  True/False.
    Every dataset listed in KAGGLE_IIRS_DATASET is searched in turn."""
    pid = product.get("product_id") or ""
    for slug in _slugs("KAGGLE_IIRS_DATASET"):
        try:
            if _fetch_member(slug, pid, "cube", out_qub):
                return True
        except Exception as exc:                         # noqa: BLE001
            print("kfetch: cube fetch failed on %s (%s)"
                  % (slug, str(exc)[:140]))
    return False

