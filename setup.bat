@echo off
REM ============================================================
REM  setup.bat  —  MLOps Demo Setup for Windows
REM  Double-click this file to set up the project automatically
REM ============================================================

echo.
echo ============================================================
echo   MLOps Demo — Auto Setup Script (Windows)
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed!
    echo         Go to https://python.org and install Python 3.11
    echo         Make sure to tick "Add to PATH" during install!
    pause
    exit /b 1
)

echo [1/3] Python found:
python --version

REM Create virtual environment
echo.
echo [2/3] Creating virtual environment...
python -m venv venv

REM Activate and install
echo.
echo [3/3] Installing packages (this takes 1-2 minutes)...
call venv\Scripts\activate.bat
pip install --quiet -r requirements.txt

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo NEXT STEPS (run in order):
echo.
echo   STEP 1: Start MLflow server
echo           In this window, run:
echo           mlflow server --host 127.0.0.1 --port 5000
echo.
echo   STEP 2: Open a NEW terminal window, activate venv, train:
echo           venv\Scripts\activate
echo           python train.py
echo.
echo   STEP 3: Open ANOTHER new terminal, start the API:
echo           venv\Scripts\activate
echo           uvicorn app:app --reload
echo.
echo   STEP 4: Open browser and go to:
echo           http://127.0.0.1:8000/docs   (API Swagger UI)
echo           http://127.0.0.1:5000        (MLflow UI)
echo.
pause
