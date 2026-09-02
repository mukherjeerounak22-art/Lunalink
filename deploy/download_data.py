"""Boot-time data restore for the SIH26166 container.

Pulls the ~29 GB mission-data snapshot (data/reference rasters, processed
scenes, demo uploads) from a Hugging Face DATASET repo into /app/data at
container start:

    HF_DATA_REPO=<username>/sih26166-data python deploy/download_data.py

Resilience contract (never kill the container over data):
- missing env / unreachable repo / partial failure -> log a clear warning
  and exit 0 so the Space still boots on the small committed caches
  (labels + browse thumbnails + processed scenes) in degraded mode
- a /app/data/.restored marker skips the download entirely on restart
  (free Spaces have ephemeral disks, so the marker usually will not
  survive a restart - the re-pull is the accepted trade-off)
- snapshot_download resumes partial transfers

Kaggle alternative: set KAGGLE_USERNAME + KAGGLE_KEY secrets and
KAGGLE_DATA_SLUG=<owner>/<dataset> to pull the same snapshot from a
Kaggle dataset instead of HF.
"""
import os
import sys
import time

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
MARKER = os.path.join(DATA_DIR, ".restored")
HF_REPO = os.environ.get("HF_DATA_REPO", "").strip()
KAGGLE_SLUG = os.environ.get("KAGGLE_DATA_SLUG", "").strip()


def already_restored() -> bool:
    if not os.path.exists(MARKER):
        return False
    # marker is only valid if the heavy rasters actually came along
    checks = [
        os.path.join(DATA_DIR, "reference", "tmc", "_library.json"),
        os.path.join(DATA_DIR, "reference", "iirs", "_library.json"),
    ]
    ok = all(os.path.exists(p) for p in checks)
    print("[data] .restored marker found, rasters present: %s" % ok)
    return ok


def restore_hf() -> bool:
    from huggingface_hub import snapshot_download
    t0 = time.time()
    print("[data] pulling %s -> %s (resume-capable, be patient on first "
          "boot)" % (HF_REPO, DATA_DIR))
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=DATA_DIR,
        max_workers=4,
    )
    print("[data] HF snapshot complete in %.0f s" % (time.time() - t0))
    return True


def restore_kaggle() -> bool:
    # kagglehub is optional in the image; import lazily
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "kagglehub"], check=False)
    import kagglehub
    t0 = time.time()
    path = kagglehub.dataset_download(KAGGLE_SLUG)
    print("[data] Kaggle dataset at %s (%.0f s)" % (path, time.time() - t0))
    # kagglehub extracts to its own cache - move contents into DATA_DIR
    import shutil
    os.makedirs(DATA_DIR, exist_ok=True)
    for entry in os.listdir(path):
        src = os.path.join(path, entry)
        dst = os.path.join(DATA_DIR, entry)
        if not os.path.exists(dst):
            shutil.move(src, dst)
    return True


def mark_restored():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MARKER, "w") as f:
        f.write("restored %s\n" % time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()))


def main():
    if already_restored():
        return 0
    try:
        if HF_REPO:
            restore_hf()
        elif KAGGLE_SLUG and os.environ.get("KAGGLE_USERNAME") \
                and os.environ.get("KAGGLE_KEY"):
            restore_kaggle()
        else:
            print("[data] no HF_DATA_REPO / KAGGLE_DATA_SLUG configured - "
                  "booting DEGRADED on the committed small caches "
                  "(browse-level auto-selection only; METRIC/MINERALS "
                  "layers need the full snapshot)")
            return 0
        # sanity: the two library indexes must exist for full function
        for p in ("reference/tmc/_library.json",
                  "reference/iirs/_library.json"):
            if not os.path.exists(os.path.join(DATA_DIR, p)):
                print("[data] WARNING: %s missing after restore - check "
                      "the dataset layout (top level must BE the contents "
                      "of data/)" % p)
        mark_restored()
        return 0
    except Exception as exc:                                  # noqa: BLE001
        print("[data] RESTORE FAILED (%s) - booting DEGRADED; the Space "
              "is up but METRIC/MINERALS layers will be unavailable until "
              "the dataset is reachable" % exc)
        return 0


if __name__ == "__main__":
    sys.exit(main())
