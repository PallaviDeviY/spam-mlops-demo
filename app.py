"""
app.py
======
MLOps Demo — FastAPI server that serves predictions from the trained model.

What this does:
  - Loads spam_model.pkl once at startup
  - Exposes GET  /           → health check
  - Exposes GET  /info       → model information
  - Exposes POST /predict    → classify one email as spam or ham
  - Exposes POST /predict/batch → classify multiple emails at once

Run with:
  uvicorn app:app --reload

Then open:
  http://127.0.0.1:8000/docs  → interactive Swagger UI (try the API here!)
  http://127.0.0.1:8000       → health check
"""

import time
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# Create the FastAPI application
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="📧 Spam Detector API",
    description="""
## MLOps Demo — Email Spam Classifier

This API classifies email text as **SPAM** or **Not Spam** using a
trained Machine Learning model (TF-IDF + Logistic Regression).

### How to use
1. Go to the `/predict` endpoint below
2. Click **Try it out**
3. Replace the example text with any email subject/body
4. Click **Execute** to see the prediction

### Try these examples
- `"WINNER!! You have been selected to receive a FREE prize!"` → SPAM
- `"Hey, are you coming to the lecture tomorrow at 10am?"` → Not Spam
- `"URGENT: Your account will be suspended! Click here NOW"` → SPAM
- `"Can you send me the notes from today's class?"` → Not Spam
""",
    version="1.0.0",
)

# ──────────────────────────────────────────────────────────────────────────────
# Load the trained model once when the server starts
# (Not inside a function — this runs at import time)
# ──────────────────────────────────────────────────────────────────────────────
print("━" * 50)
print("  Loading spam_model.pkl...")
try:
    model = joblib.load("spam_model.pkl")
    print("  ✓ Model loaded successfully!")
    print("  ✓ Server is ready to serve predictions.")
except FileNotFoundError:
    print("  ✗ ERROR: spam_model.pkl not found!")
    print("  ✗ Please run 'python train.py' first.")
    print("  ✗ The server will start but /predict will fail.")
    model = None
print("━" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# Request and Response schemas (Pydantic validates these automatically)
# ──────────────────────────────────────────────────────────────────────────────
class EmailRequest(BaseModel):
    """A single email to classify."""
    text: str = Field(
        ...,
        min_length=1,
        description="The email text (subject + body) to classify",
        examples=["WINNER!! You have been selected to receive a FREE prize worth 1000!"],
    )


class BatchEmailRequest(BaseModel):
    """Multiple emails to classify at once."""
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of email texts to classify (max 100)",
        examples=[["FREE prize! Call now!", "Meeting at 3pm today"]],
    )


class PredictionResponse(BaseModel):
    """The result of classifying one email."""
    text:       str
    prediction: str    # "spam" or "ham"
    is_spam:    bool
    confidence: float  # probability of being spam (0.0 to 1.0)
    verdict:    str    # "SPAM" or "Not Spam"
    latency_ms: float  # how long the prediction took


class BatchPredictionResponse(BaseModel):
    """Results of classifying multiple emails."""
    results:    List[PredictionResponse]
    total:      int
    spam_count: int
    ham_count:  int


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint 1: Health Check (GET /)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health Check", tags=["Status"])
def root():
    """
    Check if the API is running.
    Open this URL in your browser to confirm the server is alive.
    """
    return {
        "status":  "✓ Spam Detector API is running!",
        "version": "1.0.0",
        "docs":    "Visit http://127.0.0.1:8000/docs for interactive documentation",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint 2: Model Info (GET /info)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/info", summary="Model Information", tags=["Status"])
def info():
    """
    Get information about the currently loaded model.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train.py' first.",
        )

    tfidf_step = model.named_steps["tfidf"]
    model_step = model.named_steps["model"]

    return {
        "model_type":     "TF-IDF + Logistic Regression Pipeline",
        "vectorizer":     {
            "name":         "TfidfVectorizer",
            "max_features": tfidf_step.max_features,
            "vocabulary_size": len(tfidf_step.vocabulary_) if hasattr(tfidf_step, "vocabulary_") else "not fitted",
        },
        "classifier": {
            "name": "LogisticRegression",
            "C":    model_step.C,
        },
        "classes": list(model_step.classes_),
        "model_file": "spam_model.pkl",
        "status": "ready",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint 3: Predict One Email (POST /predict)
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify One Email",
    tags=["Prediction"],
)
def predict(email: EmailRequest):
    """
    Classify a single email as **spam** or **ham** (not spam).

    Send a POST request with JSON body:
    ```json
    {"text": "Your email text here"}
    ```

    Returns:
    - **prediction**: "spam" or "ham"
    - **is_spam**: true or false
    - **confidence**: probability of being spam (0.0 = definitely not spam, 1.0 = definitely spam)
    - **verdict**: human-readable label "SPAM" or "Not Spam"
    - **latency_ms**: how fast the prediction was made (milliseconds)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train.py' first.",
        )

    start_time = time.time()

    # Run the model
    text       = email.text
    prediction = model.predict([text])[0]          # "spam" or "ham"
    proba      = model.predict_proba([text])[0]    # [P(ham), P(spam)]

    # predict_proba returns probabilities for each class in alphabetical order
    # classes are ["ham", "spam"] → index 0 = ham, index 1 = spam
    spam_confidence = float(proba[1])

    latency_ms = (time.time() - start_time) * 1000

    return PredictionResponse(
        text       = text,
        prediction = prediction,
        is_spam    = (prediction == "spam"),
        confidence = round(spam_confidence, 4),
        verdict    = "SPAM" if prediction == "spam" else "Not Spam",
        latency_ms = round(latency_ms, 2),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint 4: Predict Batch (POST /predict/batch)
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Classify Multiple Emails",
    tags=["Prediction"],
)
def predict_batch(request: BatchEmailRequest):
    """
    Classify multiple emails in a single request.

    Send a POST request with JSON body:
    ```json
    {"texts": ["email one", "email two", "email three"]}
    ```

    Useful for:
    - Processing a folder of emails at once
    - Bulk spam filtering
    - Testing many examples quickly
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train.py' first.",
        )

    results = []
    for text in request.texts:
        start_time  = time.time()
        prediction  = model.predict([text])[0]
        proba       = model.predict_proba([text])[0]
        spam_conf   = float(proba[1])
        latency_ms  = (time.time() - start_time) * 1000

        results.append(PredictionResponse(
            text       = text,
            prediction = prediction,
            is_spam    = (prediction == "spam"),
            confidence = round(spam_conf, 4),
            verdict    = "SPAM" if prediction == "spam" else "Not Spam",
            latency_ms = round(latency_ms, 2),
        ))

    spam_count = sum(1 for r in results if r.is_spam)

    return BatchPredictionResponse(
        results    = results,
        total      = len(results),
        spam_count = spam_count,
        ham_count  = len(results) - spam_count,
    )
