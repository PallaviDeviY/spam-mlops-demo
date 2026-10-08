# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile
# MLOps Demo — Spam Detector API
#
# This file tells Docker how to build a container image for our spam API.
# Each instruction creates a "layer" in the image.
#
# Build:  docker build -t spam-api:v1 .
# Run:    docker run -p 8000:8000 spam-api:v1
# ──────────────────────────────────────────────────────────────────────────────

# ── Layer 1: Start from the official Python 3.11 slim image ──────────────────
# 'slim' = smaller image, just Python with no extras
FROM python:3.11-slim

# ── Layer 2: Set working directory inside the container ──────────────────────
# All subsequent commands will run from /app inside the container
WORKDIR /app

# ── Layer 3: Copy requirements.txt first (cache trick!) ──────────────────────
# Docker caches each layer. If requirements.txt hasn't changed,
# it won't re-run pip install even if your code changed.
# Always copy requirements BEFORE code for faster rebuilds.
COPY requirements.txt .

# ── Layer 4: Install Python packages ─────────────────────────────────────────
# --no-cache-dir keeps the image smaller (no pip cache stored)
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 5: Copy your project files into the container ──────────────────────
COPY emails.csv .
COPY train.py .
COPY app.py .

# ── Layer 6: Train the model during image build ───────────────────────────────
# The trained spam_model.pkl will be baked into the image.
# Note: MLflow logging is skipped here because no MLflow server is running.
# We train without MLflow tracking for the Docker build.
RUN python -c "
import pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('emails.csv')
X, y = df['text'].values, df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('model', LogisticRegression(C=1.0, max_iter=1000)),
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'spam_model.pkl')
print('Model trained and saved.')
"

# ── Layer 7: Create a non-root user (security best practice) ─────────────────
# Never run production containers as root!
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Layer 8: Document which port the app listens on ──────────────────────────
# This is just documentation — you still need -p 8000:8000 in docker run
EXPOSE 8000

# ── Layer 9: Health check — Docker will call this every 30s ──────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# ── Layer 10: Command to run when the container starts ───────────────────────
# --host 0.0.0.0 = accept connections from outside the container
# --port 8000    = listen on port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
