@echo off
setlocal enabledelayedexpansion

echo [1/4] Starting Docker containers...
cd /d "%~dp0"
docker compose up -d
if errorlevel 1 ( echo Docker failed & pause & exit /b 1 )

echo [2/4] Starting ngrok tunnels...
start "ngrok" ngrok start --all --config ngrok.yml

echo Waiting for ngrok API to be ready...
:wait_ngrok
curl -s http://127.0.0.1:4040/api/tunnels >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_ngrok
)

echo [3/4] Fetching ngrok tunnel URLs...
:fetch_urls
for /f "tokens=*" %%u in ('powershell -NoProfile -Command "$t=(Invoke-RestMethod http://127.0.0.1:4040/api/tunnels).tunnels; $fa=($t|?{$_.config.addr -like '*8000*'}).public_url; $fl=($t|?{$_.config.addr -like '*5000*'}).public_url; $fe=($t|?{$_.config.addr -like '*4200*'}).public_url; if($fa -and $fl -and $fe){Write-Output \"$fa|$fl|$fe\"}else{Write-Output ''}"') do set URLS=%%u

if "!URLS!"=="" (
    timeout /t 2 /nobreak >nul
    goto fetch_urls
)

for /f "tokens=1,2,3 delims=|" %%a in ("!URLS!") do (
    set FASTAPI_URL=%%a
    set FLASK_URL=%%b
    set FRONTEND_URL=%%c
)

echo FastAPI  : !FASTAPI_URL!
echo Flask    : !FLASK_URL!
echo Frontend : !FRONTEND_URL!

echo [4/4] Updating config.json...
powershell -NoProfile -Command "Set-Content -Path 'eventzilla-front\public\config.json' -Value ('{\"fastapiUrl\":\"!FASTAPI_URL!\",\"flaskUrl\":\"!FLASK_URL!\"}')"

echo Waiting for frontend container...
:wait_frontend
docker exec eventzilla-frontend ls /usr/share/nginx/html >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_frontend
)

docker cp eventzilla-front\public\config.json eventzilla-frontend:/usr/share/nginx/html/config.json
echo Pushed config.json to container.

echo.
echo ============================================
echo  App is live at: !FRONTEND_URL!
echo ============================================
start "" "!FRONTEND_URL!"
pause
