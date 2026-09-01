# =====================================================================
# start_demo_tunnel.ps1 — laptop-backend demo launcher (SIH26166)
#
# Starts:  1) the FastAPI backend on http://localhost:8000
#          2) a FREE cloudflared HTTPS tunnel (no signup, no account)
# Then prints the exact URL to open:
#          https://<your-vercel-url>/?api=<tunnel-url>
#
# Why the tunnel: the Vercel page is HTTPS, and browsers block HTTPS
# pages from calling plain-HTTP backends (mixed content). The tunnel
# gives localhost:8000 a public HTTPS address — no other changes.
#
# One-time install (only if you don't have it):
#     winget install Cloudflare.cloudflared
#
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File start_demo_tunnel.ps1
# Stop everything after the demo:  Get-Process python, cloudflared | Stop-Process -Force
# =====================================================================

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if (-not (Test-Path "$root\backend\main.py")) {
    Write-Host "Run this from the repo root (main.py not found under backend\)." -ForegroundColor Red
    exit 1
}

# --- 1. backend ------------------------------------------------------
Write-Host "[1/3] starting backend (uvicorn :8000) ..." -ForegroundColor Cyan
Start-Process -WindowStyle Hidden python `
    -ArgumentList '-m','uvicorn','main:app','--host','0.0.0.0','--port','8000' `
    -WorkingDirectory "$root\backend"

# --- 2. tunnel -------------------------------------------------------
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared not found. Install it first:" -ForegroundColor Red
    Write-Host "    winget install Cloudflare.cloudflared"
    Write-Host "(backend is running locally at http://127.0.0.1:8000 in the meantime)"
    exit 1
}
Write-Host "[2/3] starting cloudflared quick tunnel (no signup needed) ..." -ForegroundColor Cyan
$log = Join-Path $env:TEMP "sih26166_tunnel.log"
Remove-Item $log -ErrorAction SilentlyContinue
Start-Process -FilePath cloudflared `
    -ArgumentList 'tunnel','--url','http://localhost:8000',"--logfile",$log `
    -WindowStyle Hidden

# --- 3. report -------------------------------------------------------
Write-Host "[3/3] waiting for the tunnel URL ..." -ForegroundColor Cyan
$url = $null
foreach ($i in 1..45) {
    Start-Sleep -Seconds 1
    if (Test-Path $log) {
        $m = Select-String -Path $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' |
             Select-Object -First 1
        if ($m) { $url = $m.Matches[0].Value; break }
    }
}
if (-not $url) {
    Write-Host "Tunnel URL not found after 45 s — check $log" -ForegroundColor Red
    exit 1
}

try { $h = (Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 60).Content } 
catch { $h = "health check failed: $($_.Exception.Message)" }

Write-Host ""
Write-Host "================ DEMO IS READY ================" -ForegroundColor Green
Write-Host "Backend (tunnel) : $url"
Write-Host "Health           : $h"
Write-Host ""
Write-Host "Open this in the browser (replace <vercel> with your Vercel URL):" -ForegroundColor Yellow
Write-Host "    https://<vercel>/?api=$url"
Write-Host ""
Write-Host "Local fallback (same backend, no Vercel): http://127.0.0.1:8000"
Write-Host "Keep this window open during the demo."
Write-Host "Stop after:  Get-Process python, cloudflared | Stop-Process -Force"
Write-Host "===============================================" -ForegroundColor Green
