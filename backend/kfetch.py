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
    return (u, k) if u and k else None

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


def _fetch_member(slug, pid, kind, out_path):
    date = _date_of(pid)
    owner, dslug = slug.split("/", 1)
    last_err = None
    for cand in _candidate_paths(pid, date, kind):
        tmp = out_path + ".dl"
        try:
            if not download_dataset_file(owner, dslug, cand, tmp):
                last_err = "HTTP failure"
                continue
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
    success; False (and the caller's honest error) on any failure."""
    pid = product.get("product_id") or ""
    slug = _slug("KAGGLE_TMC_DATASET")
    if not slug:
        return False
    try:
        return _fetch_member(slug, pid, "dtm", out_tif)
    except Exception as exc:                             # noqa: BLE001
        print("kfetch: DTM fetch failed (%s)" % str(exc)[:140])
        return False


def ensure_cube(product, out_qub):
    """cube.qub for an IIRS product, fetched on demand.  True/False."""
    pid = product.get("product_id") or ""
    slug = _slug("KAGGLE_IIRS_DATASET")
    if not slug:
        return False
    try:
        return _fetch_member(slug, pid, "cube", out_qub)
    except Exception as exc:                             # noqa: BLE001
        print("kfetch: cube fetch failed (%s)" % str(exc)[:140])
        return False

