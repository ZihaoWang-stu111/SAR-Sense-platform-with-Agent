@echo off
echo ========================================
echo    SAR-Sense API Server (FastAPI)
echo ========================================
echo.
echo Starting server...
echo Access at: http://localhost:5000
echo API Docs at: http://localhost:5000/docs
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "%~dp0"
conda activate rag_env_backup
python api_server_fastapi.py

pause
