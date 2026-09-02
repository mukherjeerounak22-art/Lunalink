# Hosting SIH26166 online — the "share one link" pathway

Goal: nothing stored on your laptop; one public URL serves the backend,
API, ONNX model and frontend, with the full 29 GB mission data attached.

## Why this architecture (the constraint that decides everything)

| Host (free tier) | Disk | Verdict |
|---|---|---|
| Render free | 512 MB, ephemeral | Cannot hold the data |
| Fly.io free | 3 GB volume | Cannot hold the data |
| Railway trial | 100 MB–5 GB | Cannot hold the data |
| **HF Space (Docker, cpu-basic)** | **2 vCPU / 16 GB RAM / ~50 GB ephemeral** | **Fits** |
| HF dataset repo / Kaggle dataset | tens of GB, free, permanent | **Home for the data** |

So the data and the app are separated:
- **Data layer** → one public **Hugging Face dataset repo** (free, permanent).
- **App layer** → one **HF Space** (Docker) that pulls the dataset at boot
  and serves *everything else* — FastAPI, ONNX, frontend — same-origin on
  one port. The Space URL is the only link you share.

## Step 1 — upload the data snapshot (once, ~29 GB)

```powershell
pip install -U "huggingface_hub[cli,hf_transfer]"
huggingface-cli login                      # your HF token (write access)
huggingface-cli repo create sih26166-data --repo-type dataset --public
# upload the ENTIRE data/ directory (top level of the repo = contents of data/):
hf_transfer=1 huggingface-cli upload <your-username>/sih26166-data `
    "c:\Users\user\Downloads\SIH\data" data --repo-type dataset
```

Notes:
- top level of the repo must BE the contents of `data/` (the download
  script sanity-checks `reference/tmc/_library.json`)
- public is important: a public dataset downloads anonymously in the
  Space, no token juggling; the data is public ISRO/NASA science data
- this runs for a while (29 GB) — let it finish, then verify by opening
  the dataset page in a browser

Kaggle alternative: create a Kaggle dataset with the same `data/`
contents and set `KAGGLE_DATA_SLUG=<owner>/<slug>` + `KAGGLE_USERNAME` +
`KAGGLE_KEY` in the Space instead (the script supports both).

## Step 2 — create the Space (2 minutes)

1. huggingface.co → New Space → name `sih26166` → SDK: **Docker**
   (Blank template) → **Public** (private Spaces also work on free).
2. Upload the repo contents (excluding `data/` — it's not needed in the
   Space; `deploy/download_data.py` and `deploy/Dockerfile` ARE needed —
   point the Space's Dockerfile path at `deploy/Dockerfile` in Space
   settings, or copy it to the repo root).
   ```powershell
   git clone https://huggingface.co/spaces/<your-username>/sih26166
   # copy the project files in (robocopy mirrors, skips data/):
   robocopy "c:\Users\user\Downloads\SIH" .\sih26166 /MIR ^
       /XD data .git __pycache__ node_modules
   # the Space builds from Dockerfile at the repo ROOT:
   Copy-Item .\sih26166\deploy\Dockerfile .\sih26166\Dockerfile
   cd sih26166; git add -A; git commit -m "deploy"; git push
   ```

## Step 3 — configure the Space (settings → Variables and secrets)

| Name | Value |
|---|---|
| `HF_DATA_REPO` | `<your-username>/sih26166-data` |
| `SENTRY_DSN_BACKEND` | your backend DSN (same as local .env) |
| `GOOGLE_API_KEY` | Gemini key (narrator) |
| `UPSTASH_REDIS_REST_URL` / `_TOKEN` | cache + rate limiter |
| `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` | persistence |

Secrets, not plain variables, for anything key-shaped.

## Step 4 — share the link

`https://<your-username>-sih26166.hf.space` — backend, API, ONNX,
frontend, all same-origin. No `?api=` parameter needed.

## Behaviour & trade-offs (read once)

- **First boot**: builds (~3 min) then downloads 29 GB (~10–40 min —
  watch the Space logs; the UI is up during/after either way).
- **Restarts / rebuilds** re-download (free Spaces have ephemeral disks).
  Fix if it ever annoys you: paid persistent storage (~$5/mo), or keep
  the dataset small by trimming `lro_nac` to the 3–4 strips you match.
- **Degraded mode**: if the dataset is unreachable, the Space STILL boots
  and serves everything except METRIC/MINERALS rasters (the committed
  label/browse caches keep multi-instrument auto-selection alive).
- Public Space = public code. Fine for a SIH demo; make the Space private
  if you'd rather (downloads still work — the dataset stays public).

## Step 5 — free your laptop

Once the Space is verified green and the dataset page shows all files:

```powershell
Remove-Item "c:\Users\user\Downloads\SIH\data" -Recurse -Force
```

→ ~29 GB back (≈59 GB free). From then on the laptop only holds the
few-MB code repo; the internet is your demo machine.

## Alternatives (when this path doesn't fit)

- **Render free + the committed small caches** (`render.yaml` in the
  repo): lite demo — SFS layers + browse-level auto-selection work,
  METRIC/MINERALS don't; sleeps after 15 min idle. Good backup link.
- **Google Colab + cloudflared**: free, but the URL changes every run
  and sessions die — demo-only, never a shareable link.
- **Railway / Fly.io / Render paid**: simplest long-term home with a
  persistent volume if you ever want zero cold-starts.
