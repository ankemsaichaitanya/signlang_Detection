# Real-time Sign Language Detection - Run script
Set-Location $PSScriptRoot

# Activate venv
& ".\venv\Scripts\Activate.ps1"

# Install deps if missing (run once)
pip install mediapipe opencv-python ultralytics numpy --quiet 2>$null

# Run detector (press Q to quit)
python realtime.py