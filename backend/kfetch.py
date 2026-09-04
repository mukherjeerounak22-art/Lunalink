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
import zipfile


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
    import kagglehub
    date = _date_of(pid)
    last_err = None
    for cand in _candidate_paths(pid, date, kind):
        try:
            zpath = kagglehub.dataset_download(slug, path=cand)
        except Exception as exc:                         # noqa: BLE001
            last_err = str(exc)[:140]
            continue
        return _extract_member(zpath, os.path.basename(cand), out_path)
    if last_err:
        # full-dataset fallback (layout unknown) - walk for the member
        try:
            dpath = kagglehub.dataset_download(slug)
            suffix = ".tif" if kind == "dtm" else ".qub"
            for root, _, files in os.walk(dpath):
                for fn in files:
                    if fn.lower() == (pid + suffix).lower():
                        return _extract_member(os.path.join(root, fn),
                                               fn, out_path)
        except Exception as exc:                         # noqa: BLE001
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

