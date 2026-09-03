# Deploying Your App on Oracle Cloud Always Free (SIH Runbook)

Goal: a permanent, always-on server holding your backend + ONNX model + ~30GB dataset, with your frontend on Vercel/Cloudflare Pages talking to it, and Supabase/Gemini wired in via env vars.

---

## ⭐ Part 0 — EXACT values for THIS project (read first; the generic parts below still apply)

Everything below is the tailored version of the generic steps — use these
exact commands/values for SIH26166.

### 0.1 What actually needs to reach the VM
- **Code**: everything except `data/` and `.env` is on GitHub → on the VM
  just `git clone https://github.com/mukherjeerounak22-art/Lunalink.git`.
  That clone already includes the frontend, the ONNX model
  (`backend/models/descriptor.onnx`) and the small committed reference
  caches.
- **Data**: `data/` (~29 GB: TMC dtm.tif, IIRS cube.qub, LRO NAC strips,
  processed scenes, demo_upload) is gitignored → upload it separately
  (Part 2-SIH below, Windows-specific commands).
- **Secrets**: never upload — create `backend/.env` on the VM (0.4).

### 0.2 The app serves its OWN frontend — Vercel is optional
`backend/main.py` mounts `frontend/` with `html=True`, and
`frontend/config.js` defaults to same-origin (`API_BASE = ""`). So the
simplest permanent setup is **ONE link from the VM**:
`http://<vm-ip>` (or `https://api.your-domain.com` behind Cloudflare).
You do NOT need Vercel for the site to work. If you also want to keep
your existing `lunalink.vercel.app`, nothing secret is set there — just
open `https://lunalink.vercel.app/?api=https://api.your-domain.com`
(the `?api=` param overrides the baked-in base; or bake
`window.API_BASE = "https://api.your-domain.com"` into config.js and
redeploy). **No Vercel secrets are ever needed. Supabase/Gemini keys
live ONLY in the VM's `backend/.env`.**

### 0.3 Run + Nginx commands for a FastAPI app with an `app-dir`
The generic Part 3.5 says "adjust main:app" — here is the exact line for
this repo (uvicorn must see `main.py` inside `backend/`):
```ini
ExecStart=/home/ubuntu/app/venv/bin/uvicorn main:app --app-dir /home/ubuntu/app/backend --host 0.0.0.0 --port 8000
WorkingDirectory=/home/ubuntu/app
```
(If you `cd /home/ubuntu/app/backend` in ExecStart instead, drop
`--app-dir`.) Nginx MUST carry these two lines or big product uploads
(`.zip`/`.tar` up to several GB) and slow first `/layers` calls
(large-DTM extraction takes minutes) will fail with 413/504:
```nginx
client_max_body_size 5G;
proxy_read_timeout 1800s;
proxy_send_timeout 1800s;
```

### 0.4 Secrets — create `/home/ubuntu/app/backend/.env`
The backend loads `.env` from the **backend folder** (`integrations.py`
reads `BACKEND_DIR/.env`), not the repo root. Copy the values from
wherever you keep them (your local notes / the machine you demo from):
```ini
SENTRY_DSN_BACKEND=...
GOOGLE_API_KEY=...                     # Gemini narrator
UPSTASH_REDIS_REST_URL=...
UPSTASH_REDIS_REST_TOKEN=...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
```
All six are OPTIONAL — the app runs fully without any of them (TIER 0);
each key just enables its integration (narrator, cache, persistence,
error tracking). `chmod 600 backend/.env` on the VM. The frontend's
browser Sentry DSN is already baked into `frontend/config.js` — nothing
to do there.

### 0.5 Data upload from WINDOWS (Part 2 assumes Linux rsync)
Windows has no rsync by default. Two good options:
- **tar + scp (recommended, single fast stream):**
  ```powershell
  cd c:\Users\user\Downloads\SIH
  tar -cf data.tar data
  scp -i your-key.pem data.tar ubuntu@<vm-ip>:/home/ubuntu/
  # on the VM:
  mkdir -p /home/ubuntu/app && tar -xf /home/ubuntu/data.tar -C /home/ubuntu/app
  ```
- **WSL rsync (resumable if your connection drops):**
  `wsl rsync -avz --progress -e "ssh -i /mnt/c/path/to/key.pem" /mnt/c/Users/user/Downloads/SIH/data/ ubuntu@<vm-ip>:/home/ubuntu/app/data/`
Either way, run it in chunks if your connection is flaky — the VM
clones the code in seconds, only `data/` is the long haul.

### 0.6 Sanity checks specific to this app
```bash
curl http://127.0.0.1:8000/health                      # {"status":"ok"}
curl http://127.0.0.1:8000/craters | head -c 300       # 5 scenes
curl -X POST http://127.0.0.1:8000/layers/demo_tmc     # first call slow
```
Then from outside: `http://<vm-ip>/` should render the full UI (scene
dropdown → match → 3D terrain → layers). The large rasters live under
`/home/ubuntu/app/data/reference/` — check `df -h` (you want ≥15 GB
free after the upload).

---

## Part 1 — Account & VM setup

### 1.1 Sign up
1. Go to oracle.com/cloud/free and create an account. You'll need a valid card for identity verification — you won't be charged as long as you stay inside Always Free limits.
2. Pick a **Home Region** carefully — you cannot change it later without a new account. If ARM capacity is tight in your first-choice region, this matters for step 1.3.

### 1.2 Create the VM
In the OCI Console: **Menu → Compute → Instances → Create Instance**

- **Name**: something like `sih-backend`
- **Image**: Ubuntu 22.04 (or 24.04 if offered) — Canonical's official image, ARM (Aarch64) build
- **Shape**: click "Change shape" → **Ampere → VM.Standard.A1.Flex** → set **2 OCPUs / 12 GB RAM** (this is the full current free allowance — don't exceed it or it'll bill/get reclaimed)
- **Boot volume**: bump this up to ~60-80GB (still within the 200GB total free block storage pool) so your dataset + OS + app all fit on one volume — simpler than managing a separate attached volume
- **SSH keys**: generate a new key pair in the console and download the private key (or paste your own public key if you already have one) — you'll need this to log in
- Leave networking on the default VCN/subnet it offers to create

Click **Create**. It takes a couple minutes to provision.

> **If you get an "Out of capacity" error**: this is the most common Oracle free-tier hiccup. Just retry — sometimes every few minutes, sometimes it takes a few tries across different Availability Domains (there's a dropdown for this in the shape config). Don't be surprised if this takes a few attempts; it's a known quirk, not something you're doing wrong.

### 1.3 Open the right ports
By default only SSH (22) is open. You need to open 80 and 443 too.

In the Console: **Networking → Virtual Cloud Networks → (your VCN) → Security Lists → Default Security List → Add Ingress Rules**

Add two rules (Source CIDR `0.0.0.0/0`, protocol TCP):
- Destination Port 80
- Destination Port 443

(Also do this on the Ubuntu firewall itself later — see 3.3.)

### 1.4 Connect
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<your-vm-public-ip>
```
Your VM's public IP is shown on the instance's detail page in the console.

---

## Part 2 — Move the dataset over

### 2.1 From your laptop, upload via rsync (resumable, handles 30GB well)
```bash
rsync -avz --progress -e "ssh -i your-key.pem" \
  /path/to/your/dataset/ \
  ubuntu@<vm-ip>:/home/ubuntu/app/data/
```
This will take a while depending on your upload speed — 30GB over a typical home connection could be a couple hours. Start this early, and `rsync` can resume if it drops.

**Faster alternative**: if your dataset is already in cloud storage somewhere (Google Drive, a bucket, Kaggle), it's often quicker to download it directly *from the VM* (Oracle's network is fast) rather than uploading from your laptop:
```bash
# example: pulling from Kaggle directly on the VM
pip install kaggle
kaggle datasets download -d <dataset-slug> -p /home/ubuntu/app/data --unzip
```

---

## Part 3 — Set up the backend

### 3.1 Install dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv nginx git
```

### 3.2 Get your code onto the VM
```bash
git clone <your-repo-url> /home/ubuntu/app
cd /home/ubuntu/app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> ONNX Runtime ships ARM64 (`aarch64`) wheels, so `pip install onnxruntime` should just work on this VM. If you hit a wheel-not-found error for any package, check that package's PyPI page for ARM support — it's rare to be missing in 2026 but worth knowing why if it happens.

### 3.3 Open the app port locally too
```bash
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

### 3.4 Set your secrets
Create an env file (don't commit this):
```bash
nano /home/ubuntu/app/.env
```
```
GEMINI_API_KEY=your_key_here
SUPABASE_URL=your_url_here
SUPABASE_KEY=your_key_here
DATA_PATH=/home/ubuntu/app/data
```

### 3.5 Run it as a persistent service (survives reboots/crashes)
```bash
sudo nano /etc/systemd/system/sih-backend.service
```
```ini
[Unit]
Description=SIH Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/app
EnvironmentFile=/home/ubuntu/app/.env
ExecStart=/home/ubuntu/app/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```
(Adjust `main:app` and the port to whatever your backend actually uses — Flask/Express commands look different, ask if you want the exact line for your framework.)

```bash
sudo systemctl daemon-reload
sudo systemctl enable sih-backend
sudo systemctl start sih-backend
sudo systemctl status sih-backend   # confirm it's running
```

---

## Part 4 — Put Nginx in front (so you're not exposing raw port 8000)

```bash
sudo nano /etc/nginx/sites-available/sih-backend
```
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
```bash
sudo ln -s /etc/nginx/sites-available/sih-backend /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Part 5 — Domain + HTTPS via Cloudflare

1. In Cloudflare DNS for your domain, add an **A record**: `api` (or whatever subdomain) → your VM's public IP, with the orange cloud (proxy) **ON**.
2. In Cloudflare, set SSL/TLS mode to **Full** (not Flexible) once you also have a cert on the origin — or simplest for a demo: set it to **Flexible** initially, which gives you HTTPS at the Cloudflare edge immediately with zero extra origin config. Good enough to get moving; switch to Full + a Let's Encrypt cert later if you have time.
3. (Optional, more correct long-term) On the VM:
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.your-domain.com
```

---

## Part 6 — Frontend

No change from before — deploy the frontend on Vercel or Cloudflare Pages, and set its API base URL env var to `https://api.your-domain.com`.

---

## Part 7 — Before demo day

- [ ] Reboot the VM once on purpose (`sudo reboot`) and confirm the backend + Nginx come back up automatically via `systemctl status`
- [ ] Hit your live API URL from a phone on mobile data (not campus wifi) to make sure it's actually publicly reachable
- [ ] Check `df -h` to confirm your dataset + app comfortably fit with room to spare
- [ ] Take a **snapshot/boot volume backup** in the OCI console once everything works, so you have a restore point if you break something the night before
- [ ] Note the VM's public IP separately in case DNS propagation is ever an issue mid-demo — you can hit the IP directly as a fallback

---

## Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| Can't SSH in | Security list port 22 not open, or wrong key file permissions (`chmod 400`) |
| Site unreachable but SSH works | Port 80/443 not open in *both* the OCI security list and `ufw` |
| 502 from Nginx | Backend service isn't running — check `systemctl status sih-backend` and `journalctl -u sih-backend -f` |
| "Out of host capacity" creating VM | Known Oracle free-tier issue — retry, try a different Availability Domain |
| ONNX install fails | Rare ARM wheel gap — check the package's PyPI ARM64 support |
