@echo off
echo Starting EventZella Backends...

set BACKEND_DIR=C:\Users\larry\Desktop\Esprit-PABI-4ERPBI6-2526-EventZella\backend

start "Flask Backend (Port 5000)" cmd /k "cd /d %BACKEND_DIR% && python app.py"
start "FastAPI Backend (Port 8000)" cmd /k "cd /d %BACKEND_DIR% && python app2.py"

echo Backends are starting in new windows.
echo Flask: http://localhost:5000
echo FastAPI: http://localhost:8000
pause
