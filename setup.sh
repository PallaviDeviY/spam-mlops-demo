#!/bin/bash
# ============================================================
#  setup.sh  —  MLOps Demo Setup for Mac / Linux
#  Run with:  bash setup.sh
# ============================================================

set -e  # exit immediately if any command fails

echo ""
echo "============================================================"
echo "  MLOps Demo — Auto Setup Script (Mac / Linux)"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 is not installed!"
    echo "        Install it with:"
    echo "          macOS:  brew install python"
    echo "          Ubuntu: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

echo "[1/3] Python found: $(python3 --version)"

# Create virtual environment
echo ""
echo "[2/3] Creating virtual environment..."
python3 -m venv venv

# Activate and install
echo ""
echo "[3/3] Installing packages (this takes 1-2 minutes)..."
source venv/bin/activate
pip install --quiet -r requirements.txt

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "NEXT STEPS (run in order):"
echo ""
echo "  STEP 1: Activate your environment (do this in EVERY new terminal)"
echo "          source venv/bin/activate"
echo ""
echo "  STEP 2: Start MLflow server (Terminal 1)"
echo "          mlflow server --host 127.0.0.1 --port 5000"
echo ""
echo "  STEP 3: Train the model (Terminal 2)"
echo "          python train.py"
echo ""
echo "  STEP 4: Start the API server (Terminal 3)"
echo "          uvicorn app:app --reload"
echo ""
echo "  STEP 5: Open in browser:"
echo "          http://127.0.0.1:8000/docs   ← API Swagger UI"
echo "          http://127.0.0.1:5000        ← MLflow UI"
echo ""
echo "  STEP 6: Run all tests"
echo "          python test_api.py"
echo ""
