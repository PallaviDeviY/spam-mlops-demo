# 📧 MLOps & API Demo — Email Spam Detector

**6th Semester · AI & ML · Dr. Ambedkar Institute of Technology**

This project builds a complete MLOps pipeline for email spam detection.
You will train an ML model, serve it as a REST API, track experiments
with MLflow, and package everything in Docker — in 45 minutes.

---

## 📁 Files in This Project

| File | What It Does |
|------|-------------|
| `emails.csv` | Training dataset — 100+ labeled spam and ham emails |
| `train.py` | Trains the ML model (TF-IDF + Logistic Regression) and logs to MLflow |
| `app.py` | FastAPI server — serves predictions through HTTP endpoints |
| `test_api.py` | Test script — sends sample emails and shows predictions |
| `requirements.txt` | Python package list |
| `Dockerfile` | Instructions to build a Docker container |
| `docker-compose.yml` | Runs API + MLflow together with one command |
| `setup.sh` | Auto-setup script for Mac/Linux |
| `setup.bat` | Auto-setup script for Windows |

---

## 🚀 Quick Start (Without Docker)

### Prerequisites
- Python 3.8 or higher — download from https://python.org
- Git (optional)

---

### Step 1 — Setup

**Mac / Linux:**
```bash
bash setup.sh
```

**Windows:**
```
Double-click setup.bat
```

**Or manually:**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

### Step 2 — Start MLflow Tracking Server

Open **Terminal 1** and run:
```bash
source venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000
```
Open your browser → **http://127.0.0.1:5000** — you should see the MLflow UI.

> Keep this terminal open. Do not close it.

---

### Step 3 — Train the Model

Open **Terminal 2** and run:
```bash
source venv/bin/activate
python train.py
```

You should see:
```
[1/5] Loading data from emails.csv...
      Loaded 105 emails total
[2/5] Preparing features and labels...
      Training set: 84 emails
      Test set:     21 emails
[3/5] Connecting to MLflow at http://127.0.0.1:5000...
[4/5] Building and training the pipeline...
[5/5] Evaluating on test set...

  RESULTS:
  Accuracy : 95.2%
  F1 Score : 0.947
  ...
  Model saved to: spam_model.pkl
```

Go back to **http://127.0.0.1:5000** — your experiment run should now be listed!

> 💡 **Try changing parameters!** Open `train.py`, change `max_features` from 5000 to 1000, save, and run again. You'll see two runs in MLflow to compare.

---

### Step 4 — Start the API Server

Open **Terminal 3** and run:
```bash
source venv/bin/activate
uvicorn app:app --reload
```

You should see:
```
Loading spam_model.pkl...
✓ Model loaded successfully!
INFO: Uvicorn running on http://127.0.0.1:8000
```

Open your browser → **http://127.0.0.1:8000/docs** — you should see the Swagger UI.

---

### Step 5 — Test the API

**Method 1: Swagger UI (easiest)**
1. Go to http://127.0.0.1:8000/docs
2. Click `POST /predict`
3. Click **Try it out**
4. Replace the text with any email
5. Click **Execute**
6. See the prediction response!

**Method 2: Python script**
```bash
python test_api.py
```

Expected output:
```
Test 3: Single Email Predictions
  Verdict    Confidence   Correct?   Email
  ─────────  ─────────    ────────   ─────────────
  SPAM       98.7%        ✓          WINNER!! You have been selected...
  Not Spam    2.1%        ✓          Hey, are you coming to the lectu...
  SPAM       96.4%        ✓          URGENT: Your account will be sus...
  Not Spam    1.9%        ✓          Can you please send me the notes...
  ...
  Score: 8/8 correct (100%)
```

**Method 3: cURL (from terminal)**
```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "WINNER! FREE prize click now!"}'
```

---

### Step 6 — Docker (Optional)

Make sure Docker Desktop is installed and running, then:

```bash
# Build the container image
docker build -t spam-api:v1 .

# Run the container
docker run -p 8000:8000 spam-api:v1

# Or run API + MLflow together
docker compose up
```

---

## 🌐 URLs

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8000 | API root (health check) |
| http://127.0.0.1:8000/docs | Interactive Swagger UI |
| http://127.0.0.1:8000/predict | POST endpoint (use Swagger or cURL) |
| http://127.0.0.1:8000/predict/batch | Batch prediction endpoint |
| http://127.0.0.1:8000/info | Model information |
| http://127.0.0.1:5000 | MLflow experiment tracking UI |

---

## 🔴 Common Errors

| Error Message | Cause | Fix |
|---------------|-------|-----|
| `ModuleNotFoundError: No module named 'fastapi'` | Package not installed | `pip install -r requirements.txt` |
| `FileNotFoundError: spam_model.pkl` | Model not trained yet | Run `python train.py` first |
| `Address already in use :::8000` | Port 8000 is busy | Stop any running uvicorn, or use `--port 8001` |
| `Connection refused 127.0.0.1:8000` | Server not running | Start `uvicorn app:app --reload` |
| `422 Unprocessable Entity` | Wrong JSON field name | Use `{"text": "..."}` not `{"email": "..."}` |
| `MLflow connection refused` | MLflow server not started | Run `mlflow server --host 127.0.0.1 --port 5000` |
| `docker: Cannot connect to Docker daemon` | Docker Desktop not open | Launch Docker Desktop first |

---

## 🧪 Experiment Ideas

1. **Change `max_features`** in `train.py`: try 500, 1000, 5000, 10000. Does accuracy improve?
2. **Change `C`** (regularization): try 0.1, 1.0, 10.0. What happens to precision vs recall?
3. **Add a new email** to `emails.csv` and retrain. Does the model learn it?
4. **Add a new endpoint** to `app.py`: `GET /stats` that returns how many predictions have been made.
5. **Try a different classifier**: change `LogisticRegression` to `MultinomialNB` from `sklearn.naive_bayes`.

---

## 📖 Key Concepts

| Concept | What It Means in This Project |
|---------|-------------------------------|
| **TF-IDF** | Converts email text into a vector of word frequency scores |
| **Logistic Regression** | Binary classifier that learns spam vs ham decision boundary |
| **Pipeline** | Chains TF-IDF + LogReg into a single reusable object |
| **MLflow** | Tracks every training run — params, metrics, model artifacts |
| **FastAPI** | Web framework that turns Python functions into HTTP endpoints |
| **Pydantic** | Validates request/response data types automatically |
| **Uvicorn** | ASGI server that runs the FastAPI application |
| **Docker** | Packages the entire app into a portable container |
| **Swagger UI** | Auto-generated interactive documentation for the API |
