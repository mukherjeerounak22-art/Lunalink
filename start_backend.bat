@echo off
setlocal EnableExtensions

rem ====================================================================
rem  SIH26166 (Lunalink) - ONE-COMMAND BACKEND LAUNCHER + POWER REVERT
rem
rem  WHERE: this file must live in the repo root:
rem            c:\Users\user\Downloads\SIH\start_backend.bat
rem  HOW:   right-click then choose Run as administrator  (or open an
rem         Administrator Command Prompt and run it - it also offers to
rem         self-elevate if you forget)
rem
rem  WHAT it does:
rem    1. cd's to the repo root automatically
rem    2. reverts power settings: lid close = SLEEP, idle 5 min = SLEEP
rem       (plugged in AND battery, Balanced AND Power saver schemes)
rem    3. starts the FastAPI backend  (uvicorn, http://127.0.0.1:8000)
rem    4. starts the free cloudflared HTTPS tunnel
rem    5. prints the exact demo URL to open:
rem demo URL: https://lunalink.vercel.app/?api=[your-tunnel-url]
rem
rem  STOP everything afterwards:
rem     taskkill /f /im python.exe & taskkill /f /im cloudflared.exe
rem ====================================================================

rem ---- 1. go to the repo root (the folder this script lives in) ----
set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
cd /d "%REPO%"
echo [1/5] Repo folder: %REPO%
if not exist "%REPO%\backend\main.py" (
  echo ERROR: backend\main.py not found next to this script.
  echo        start_backend.bat must stay in the repo root.
  pause
  exit /b 1
)

rem ---- 2. administrator check (self-elevate if forgotten) ----
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo [2/5] Not elevated - relaunching with administrator rights...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
echo [2/5] Administrator rights confirmed.

rem ---- 3. revert power settings ----
rem   lid close          = 1 (Sleep)   - AC and DC
rem   idle - sleep after 300 s (5 min) - AC and DC
rem   applied to BOTH the Balanced ("normal") and Power saver schemes
echo [3/5] Restoring power settings: lid close = SLEEP, idle 5 min = SLEEP ...
for %%S in (SCHEME_BALANCED SCHEME_MIN) do (
  powercfg /setacvalueindex %%S SUB_BUTTONS LIDACTION 1 >nul 2>&1
  powercfg /setdcvalueindex %%S SUB_BUTTONS LIDACTION 1 >nul 2>&1
  powercfg /setacvalueindex %%S SUB_SLEEP STANDBYIDLE 300 >nul 2>&1
  powercfg /setdcvalueindex %%S SUB_SLEEP STANDBYIDLE 300 >nul 2>&1
)
rem also restore the hibernate-after defaults the demo script had disabled
powercfg /change hibernate-timeout-ac 180 >nul 2>&1
powercfg /change hibernate-timeout-dc 120 >nul 2>&1
rem force the active scheme to pick the changes up
powercfg /setactive SCHEME_CURRENT >nul 2>&1
echo       done.

rem ---- 4. start the backend ----
echo [4/5] Starting backend (uvicorn on http://127.0.0.1:8000) ...
start "SIH26166 backend" cmd /k "cd /d "%REPO%\backend" && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

rem ---- 5. start the HTTPS tunnel and print the demo URL ----
echo [5/5] Starting cloudflared HTTPS tunnel ...
set "CF=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
if not exist "%CF%" set "CF=%ProgramFiles%\cloudflared\cloudflared.exe"
if not exist "%CF%" set "CF=%LOCALAPPDATA%\Microsoft\WinGet\Links\cloudflared.exe"
if not exist "%CF%" (
  echo ERROR: cloudflared not found. Install it once with:
  echo        winget install Cloudflare.cloudflared
  echo The backend IS running locally at http://127.0.0.1:8000
  pause
  exit /b 1
)
set "TLOG=%TEMP%\sih26166_tunnel.log"
if exist "%TLOG%" del "%TLOG%" >nul 2>&1
start "SIH26166 tunnel" /min "%CF%" tunnel --url http://localhost:8000 --logfile "%TLOG%"

set "TURL="
set /a TRIES=0
:waitloop
if defined TURL goto goturl
set /a TRIES+=1
if %TRIES% GEQ 75 goto nourl
timeout /t 2 /nobreak >nul
if not exist "%TLOG%" goto waitloop
for /f "usebackq tokens=*" %%L in (`findstr /c:"trycloudflare.com" "%TLOG%" 2^>nul`) do (
  for %%u in (%%L) do (
    echo %%u | findstr /b /c:"https://" >nul && set "TURL=%%u"
  )
)
goto waitloop

:goturl
echo.
echo ============================================================
echo   BACKEND READY
echo.
echo   local backend : http://127.0.0.1:8000
echo   tunnel        : %TURL%
echo.
echo   DEMO URL - open this in the browser:
echo     https://lunalink.vercel.app/?api=%TURL%
echo.
echo   STOP everything afterwards:
echo     taskkill /f /im python.exe ^& taskkill /f /im cloudflared.exe
echo ============================================================
where curl >nul 2>&1 && curl -s -m 30 "%TURL%/health" && echo.
pause
exit /b 0

:nourl
rem one last look - on a slow hotspot the tunnel may register just after the window
for /f "usebackq tokens=*" %%L in (`findstr /c:"trycloudflare.com" "%TLOG%" 2^>nul`) do (
  for %%u in (%%L) do (
    echo %%u | findstr /b /c:"https://" >nul && set "TURL=%%u"
  )
)
if defined TURL goto goturl
echo.
echo Tunnel did not report a URL within 150 s - check the log:
echo   %TLOG%
echo The backend is still running at http://127.0.0.1:8000
pause
exit /b 1