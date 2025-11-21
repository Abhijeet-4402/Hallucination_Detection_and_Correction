#!/bin/bash

echo "==================================================="
echo "Setting up and Running Hallucination Detector Locally"
echo "==================================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "[WARNING] .env file not found!"
    echo "Please create a .env file with your GEMINI_API_KEY."
    echo "You can copy .env.example to .env and edit it."
    read -p "Press enter to exit..."
    exit 1
fi

# Dependencies are assumed to be installed
echo "[INFO] Skipping dependency check as requested."

# Start Flask API in background
echo "[INFO] Starting Flask API..."
python "Frontend Code/api.py" &
API_PID=$!

# Wait a few seconds for API to initialize
sleep 5

# Start Streamlit Frontend
echo "[INFO] Starting Streamlit App..."
streamlit run "Frontend Code/app.py"

# Cleanup: Kill API when Streamlit exits
kill $API_PID
