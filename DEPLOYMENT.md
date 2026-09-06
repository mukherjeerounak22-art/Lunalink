# SIH26166 — Deployment & Integrations Guide

> **Want the whole project online with ONE shareable link (and nothing
> stored on your laptop)?** Follow **`deploy/DEPLOY_ONLINE.md`** —
> HF dataset repo for the ~29 GB data + one HF Space serving backend,
> API, ONNX and frontend same-origin. The Render/Vercel path below is
> the lighter-weight backup option.

Everything here is optional: the demo runs fully on localhost with **zero keys**
(TIER 0). Follow sections in order only for what you need.

---

## 1. Supabase setup (persistence)

1. Go to **supabase.com** → sign in → **New project**.
   - Name: `sih26166`, pick a **region near your judging venue**, set a DB
     password (save it — only needed for direct DB access, not our API path).
2. Wait ~2 min for provisioning.
3. **SQL Editor** (left sidebar) → **New query** → paste the entire contents of
   `supabase/schema.sql` from this repo → **RUN**.
   - Creates `scenes`, `jobs`, `matches`, `metrics` + RLS + the 3 storage
     buckets (`raw-tiles`, `dem-patches`, `model-weights`).
4. **Project Settings (gear icon) → API** — copy these three values:
   - `Project URL` → **SUPABASE_URL**
   - `anon public` key → frontend only (we don't use it — backend service role
     covers everything)
   - `service_role` key → **SUPABASE_SERVICE_ROLE_KEY** ⚠️ backend `.env` only,
     NEVER in the frontend or git.
5. Paste both into `backend/.env` (copy from `backend/.env.example`).
6. Restart uvicorn → run a match → check **Table Editor → matches/metrics**:
   rows appear automatically after each `/match` run.

## 2. Upstash Redis setup (job status + rate limiting)

1. Go to **upstash.com** → sign in → **Create Database**.
   - Name: `sih26166`, Type: **Regional**, region closest to the backend host.
   - Free plan is fine.
2. Open the database → **REST API** section in the dashboard sidebar.
3. Copy:
   - `UPSTASH_REDIS_REST_URL` (looks like `https://xxx.upstash.io`)
   - `UPSTASH_REDIS_REST_TOKEN` (long JWT — treat as a secret)
4. Paste into `backend/.env`, restart uvicorn.
5. Verify: run `/match/tycho`, then Upstash dashboard → **Data Browser** —
   you should see `job:tycho:status` with a JSON body.
   Rate limiting: call `/narrate/tycho` 21× in a minute → the 21st returns 429
   (the frontend shows the rate-limit message; that's a demo-able feature).

## 3. Sentry setup (error tracking, 2 separate projects)

1. Go to **sentry.io** → sign up/sign in.
2. **Create Project** → platform **Python (FastAPI)** → name `sih26166-backend`.
   Copy the DSN (`https://...ingest.sentry.io/...`) → **SENTRY_DSN_BACKEND**.
3. **Create Project** again → platform **JavaScript (browser)** → name
   `sih26166-frontend`. Copy that DSN → **VITE_SENTRY_DSN_FRONTEND** (only
   needed if you later add a browser bundle; the backend works without it).
4. Paste into `backend/.env`, restart uvicorn.
5. Verify: request `/match/nonexistent` and check the Sentry issues page.
   Breadcrumbs are added after each pipeline stage, so a silent
   coordinate-transform bug shows up as an event with full context.


## 4. Google Gemini setup (narration ONLY — never generates metrics)

1. Go to **aistudio.google.com** → sign in → **Get API key** → **Create API
   key** (free tier, no billing needed).
2. Copy → **GOOGLE_API_KEY** in `backend/.env`.
3. ⚠️ Check the current model string at `ai.google.dev/gemini-api/docs/models`
   before demo day — names change often; set via `GEMINI_MODEL` if the default
   (`gemini-2.0-flash`) is retired. Don't trust a remembered name.
4. Restart uvicorn → click **🔊 Narrate for judges** in the frontend → the
   panel shows `[Gemini] ...`. Without a key it shows `[local template]` —
   same text style, zero dependency.

## 5. GitHub (version control)

`.gitignore` already excludes `data/` (1 GB+ scenes), `*.onnx`, `.env`,
`__pycache__`, `.vercel`. Then:

```bash
cd C:\Users\user\Downloads\SIH
git init
git add .
git commit -m "SIH26166: OHRC matching pipeline + holographic frontend + integrations"

# Option A - GitHub CLI (winget install GitHub.cli, then gh auth login)
gh repo create SIH26166 --private --source=. --push

# Option B - manual: create an EMPTY repo on github.com (no readme init), then:
git remote add origin https://github.com/<your-username>/SIH26166.git
git branch -M main
git push -u origin main
```

⚠️ Before pushing, check `git status` — you must NOT see `data/`, `.env`, or
any `.onnx`. If you do, stop and fix `.gitignore` first.

## 6. Vercel (frontend) + backend — the deployed demo

The frontend is a static single page — ideal for Vercel. **Do not deploy the
backend to Vercel**: Python + OpenCV + scene data don't fit serverless cold
starts. Two backend options:

- **Option A (demo day): backend on your laptop + free cloudflared HTTPS
  tunnel** — `start_demo_tunnel.ps1` (repo root) does it in one command and
  prints `https://<vercel>/?api=<tunnel>`. The `?api=` query parameter is
  honored by `frontend/config.js`, so changing tunnel URLs never needs a
  Vercel redeploy. Full caveats: [`DEMO_INSTRUCTIONS.md` §14 Option A].
- **Option B (always-on): backend on Render** from the repo's
  `render.yaml` Blueprint, which includes the committed demo scenes and
  `descriptor.onnx`.

1. (Once) `npm i -g vercel`, then `vercel login`.
2. From the repo root: `vercel` → **Link to existing project? N** → name
   `sih26166` → keep defaults (the repo-root `vercel.json` sets
   `outputDirectory: frontend`).
3. Deploy the backend: **render.com** → New → **Blueprint** → pick this
   repo → Apply (reads `render.yaml`). Verify
   `https://<backend>.onrender.com/health` → `"learned_model_loaded": true`.
4. Edit `frontend/config.js` →
   `window.API_BASE = "https://<your-backend-host>"` → `vercel --prod`.
5. Production URL: `https://sih26166.vercel.app`. Pre-warm the backend
   before demoing — the Render free tier sleeps after ~15 min idle.

Full walkthrough + demo-day caveats (cold start, simulated second-pass
references for fresh uploads): [`DEMO_INSTRUCTIONS.md` §14](DEMO_INSTRUCTIONS.md).


### Backend hosting (Render Blueprint — free tier works)
`render.yaml` at the repo root encodes everything: **runtime python ·
rootDir `backend` · build `pip install -r requirements.txt` · start
`uvicorn main:app --host 0.0.0.0 --port $PORT` · health check `/health`**.
Optional keys (Gemini, Supabase, Upstash, Sentry) go in the Render
dashboard → Environment — never in git. The demo scenes and the trained
model are committed, so no manual data upload is needed.

## 7. ONNX descriptor integration (after Kaggle training)

1. Run the 7 Kaggle cells (chat/README section) → download `descriptor.onnx`
   from the notebook **Output** tab (~10–15 min on T4×2).
2. Place it at exactly `backend/models/descriptor.onnx`.
3. `pip install onnxruntime` (already in `backend/requirements.txt`).
4. Restart uvicorn — **auto-detected at import time**:
   - `curl localhost:8000/health` → `"learned_model_loaded": true`
   - `/match` now unions SIFT + learned candidates; `method_breakdown` shows
     `learned_model_loaded: true` + nonzero `learned_candidates`.
5. Validation gate: Cell 6 must print triplet ranking accuracy **> 0.5**
   (ideally > 0.9). Below 0.5 the descriptor hurts matching — don't ship it.

## 8. Six-point smoke test (before demo day)

1. `curl localhost:8000/health` → ok (+ learned flag if model present)
2. Frontend loads at `/`, both scenes match, hologram renders
3. Supabase → `scenes` row exists, `matches`/`metrics` rows appear per run
4. Upstash → `job:tycho:status` key visible; 21st `/narrate` call → 429
5. Sentry → trigger `/match/nonexistent`, event appears
6. Gemini → narration panel shows `[Gemini]` tag with your metrics only

Then: screen-record a backup demo video.

## 9. One-link hosting — share ONE URL for everything (backend + API + ONNX + frontend + static imagery)

The frontend is a zero-build single HTML file that the FastAPI backend
**already serves itself** (`app.mount("/", StaticFiles(directory=FRONTEND))`,
and `frontend/config.js` uses `window.API_BASE = ""` = same origin). So the
simplest shareable deployment is **one web service, one URL** — no Vercel
step, no CORS juggling, no second deploy:

**Recommended path: Render web service (the included `render.yaml`) — THIS IS LIVE**

> ✅ **Deployed:** https://sih26166-backend.onrender.com (free tier,
> Python 3.11, 512 MB). Environment variables set in the Render dashboard:
> `GOOGLE_API_KEY`, `SENTRY_DSN_BACKEND`, `UPSTASH_REDIS_REST_URL`/`_TOKEN`,
> `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `KAGGLE_USERNAME`/`KAGGLE_KEY`,
> and the dataset pointers `KAGGLE_LRO_DATASET`,
> `KAGGLE_TMC_DATASET`, `KAGGLE_IIRS_DATASET` — the latter two accept
> **comma-separated lists** of public Kaggle datasets; the on-demand fetcher
> (`backend/kfetch.py`) searches every listed dataset per product file, so
> the raw rasters can live scattered across the account's datasets. All six
> baked scenes, the ONNX model and the layer caches ship in git, so a fresh
> deploy serves the full experience immediately; only brand-new uploads
> trigger Kaggle fetches. The match pipeline is RAM-capped for the free
> tier's 512 MB (measured peak 275 MB). A GitHub Actions keep-alive
> (`.github/workflows/keep-alive.yml`, secret `WAKE_URL`) pings `/health`
> every 10 minutes so the free tier doesn't sleep mid-demo; the UI also
> auto-wakes a cold instance with a pulsing status badge.

1. Push the repo to GitHub (`origin` is already configured).
2. render.com → **New + → Blueprint** → pick the repo → **Apply**.
   `render.yaml` deploys `backend/` with `pip install -r requirements.txt`
   and `uvicorn main:app --host 0.0.0.0 --port $PORT`, health-checked at
   `/health`.
3. Wait for the build (~5–10 min). `https://sih26166-backend.onrender.com`
   now serves **everything**: the UI at `/`, the API, `/static` scene
   imagery, and the trained `backend/models/descriptor.onnx` (committed, so
   `/health` reports `learned_model_loaded: true` on a fresh deploy).
4. Optional keys (Gemini, Supabase, Upstash, Sentry) go in the Render
   dashboard → Environment — never in git.
5. Share `https://<your-service>.onrender.com`. Done.

Why this beats the two-URL Vercel+Render setup for sharing: one link is one
thing to remember, the `?api=` trick becomes unnecessary, and the page can
never point at a stale backend. (The Vercel option in §6 still works if you
want a separate pretty domain for the UI.)

**Free-tier caveats & honest limits**

- The free plan sleeps after ~15 min idle (first visitor waits ~50 s) and
  has 512 MB RAM + ephemeral disk. The committed baked scenes
  (`data/processed/*`), demo upload set, ONNX model, and the **lightweight
  ISRO TMC reference library** (`data/reference/tmc/` — browse thumbnails +
  labels only, ~21 MB, committed via `.gitignore`) all boot with the
  service, so multi-instrument auto-selection works out of the box.
- The multi-GB raw products (LRO NAC strips, TMC-2 DTM GeoTIFFs, IIRS
  cubes) are NOT deployable to a free tier — large product uploads
  (`/ingest_product_upload`) are still possible client-side but give the
  best experience locally. Small product ZIPs and images work fine.
  (Local disk note: the PRADAN bundle TARs and the IIRS product ZIP were
  deleted after extraction — the working rasters live extracted under
  `data/reference/` (`tmc/<pid>/dtm.tif`, `iirs/<pid>/cube.qub`), which
  is what the layer pipeline reads; re-download from PRADAN if you ever
  need to re-ingest *new* crops from the original bundles.)
- If you need the full raw libraries hosted too: Render paid tier with a
  persistent disk, **Railway** / **Fly.io** (volume mounts, more RAM), or
  **Hugging Face Spaces** (Docker, free 16 GB RAM container — add a
  `Dockerfile` that runs `uvicorn main:app --host 0.0.0.0 --port 7860`
  and put the big data in a Spaces dataset or an attached persistent
  storage). Same single-URL story on all of them.

## 8. Azure Container Apps — hybrid Kaggle (data) + Azure (compute), 7am–7pm scheduled

Architecture: **Kaggle stays the data plane** (~56 GB of NAC strips, TMC-2
DTMs and IIRS cubes across the account's public datasets — fetched per-file
on demand by `backend/kfetch.py`), **Azure runs only compute** (one
container: FastAPI + frontend + ONNX). The rasters never enter git or Azure
storage. Unlike App Service (which bills 24/7 for the plan), Container Apps
bills per replica-second and can scale to **zero** — this deployment runs
`min-replicas 1` from **07:00–19:00 IST** and `min-replicas 0` (≈ $0)
overnight, driven by `.github/workflows/azure-schedule.yml` (crons 01:30 /
13:30 UTC). The image ships from Docker Hub (free) via
`.github/workflows/docker-build.yml` — no ACR cost.

**One-time Cloud Shell setup** (shell.azure.com, Bash; skip any step whose
resources already exist from the earlier App Service attempt — or wipe with
`az group delete -n sih26166-rg -y`):

```bash
RG=sih26166-rg; LOC=centralindia
az group create -n $RG -l $LOC
az containerapp env create -n sih26166-env -g $RG -l $LOC

# scheduler identity (for the GitHub cron) — save appId/password/tenant
az ad sp create-for-rbac --name sih26166-scheduler --role Contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RG
az account show --query id -o tsv

# app (public Docker Hub image pushed by the build workflow; 1 GiB covers
# the measured 275 MB match peak). Fill the env list from the dashboard
# values used on Render, then:
az containerapp create -n sih26166-backend -g $RG \
  --environment sih26166-env \
  --image <DOCKERHUB_USER>/sih26166:latest \
  --ingress external --target-port 7860 \
  --min-replicas 0 --max-replicas 2 --cpu 0.5 --memory 1.0Gi \
  --env-vars \
  "GOOGLE_API_KEY=<...>" "GEMINI_MODEL=gemini-flash-lite-latest" \
  "SENTRY_DSN_BACKEND=<...>" \
  "UPSTASH_REDIS_REST_URL=<...>" "UPSTASH_REDIS_REST_TOKEN=<...>" \
  "SUPABASE_URL=<...>" "SUPABASE_SERVICE_ROLE_KEY=<...>" \
  "KAGGLE_USERNAME=rounakmukherjee22" "KAGGLE_KEY=<...>" \
  "KAGGLE_LRO_DATASET=rounakmukherjee22/lro-nac-polar" \
  "KAGGLE_TMC_DATASET=<comma-separated TMC dataset list>" \
  "KAGGLE_IIRS_DATASET=<comma-separated IIRS dataset list>"

az containerapp show -n sih26166-backend -g $RG \
  --query properties.configuration.ingress.fqdn -o tsv
```

Live URL: `https://sih26166-backend.<env-hash>.centralindia.azurecontainerapps.io`
(free managed HTTPS). **GitHub secrets required** (repo → Settings →
Secrets → Actions): `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (hub.docker.com
→ Settings → Security → New Access Token), and `AZURE_CREDS` = one JSON
`{"clientId": "<appId>", "clientSecret": "<password>", "subscriptionId":
"<sub id>", "tenantId": "<tenant>"}`. Then Actions → **build-push-dockerhub
→ Run workflow** once to publish the image, and the azure-schedule cron
drives 7am–7pm thereafter (manual override: run it with state on/off).
Overnight any visitor wakes the app in 1–3 min (all caches ship in the
image; the UI's wake badge covers the wait). Cost at 12 h/day,
0.5 vCPU/1 GiB ≈ **$14/mo after the free consumption grants** (vCPU
180 k-s + memory 360 k-GiB-s per month) → **~7 months from the $100
student credit**; Docker Hub $0; Kaggle $0. Keep the Render keep-alive
(`WAKE_URL`) pointed at **Render** so the free fallback stays warm; do not
point it at the Container App or the pinger defeats the night schedule.
Cleanup: `az group delete -n sih26166-rg -y`.



