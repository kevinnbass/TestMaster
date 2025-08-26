@echo off
REM TestMaster Dashboard Production Startup Script for Windows
REM Usage: start_production.bat

echo 🚀 Starting TestMaster Dashboard in Production Mode
echo =================================================

REM Check if virtual environment exists
if not exist "venv" (
    echo ⚠️  No virtual environment found. Creating one...
    python -m venv venv
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo 📥 Installing production dependencies...
pip install -r requirements.txt

REM Set production environment variables
set ENVIRONMENT=production
set FLASK_ENV=production
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Create log directory if it doesn't exist
if not exist "logs" mkdir logs

REM Start Gunicorn with configuration
echo 🌐 Starting Gunicorn server...
for /f %%i in ('python -c "import multiprocessing; print(multiprocessing.cpu_count() * 2 + 1)"') do set WORKERS=%%i
echo    - Workers: %WORKERS%
echo    - Binding: 0.0.0.0:5000
echo    - Configuration: gunicorn_config.py
echo.

gunicorn --config gunicorn_config.py wsgi:application