@echo off
echo ===========================================
echo   Retina Vessel Segmentation Web System
echo ===========================================
echo.
echo Starting FastAPI server...
cd %~dp0\backend
python -m uvicorn main:app --reload --port 8000
