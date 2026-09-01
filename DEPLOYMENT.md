# SIH26166 — Deployment & Integrations Guide

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


