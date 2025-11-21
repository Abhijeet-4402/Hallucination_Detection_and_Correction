@echo off
echo ===================================================
echo Setting up and Running Hallucination Detector Locally
echo ===================================================

REM Check if .env exists
if not exist .env (
    echo [WARNING] .env file not found!
    echo Please create a .env file with your GEMINI_API_KEY.
    echo You can copy .env.example to .env and edit it.
    pause
    exit /b
)

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

REM Start Flask API in a new window
echo [INFO] Starting Flask API...
start "Flask API" cmd /k "python "Frontend Code/api.py""

REM Wait a few seconds for API to initialize
timeout /t 5

REM Start Streamlit Frontend
echo [INFO] Starting Streamlit App...
streamlit run "Frontend Code/app.py"

pause
