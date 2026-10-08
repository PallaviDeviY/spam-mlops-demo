"""
train.py
========
MLOps Demo — Train an email spam classifier.

What this script does:
  1. Reads emails.csv (your labeled dataset)
  2. Splits data into train and test sets (80/20)
  3. Builds a Pipeline: TF-IDF vectorizer → Logistic Regression
  4. Logs the experiment to MLflow (params + metrics + model)
  5. Saves the trained model as spam_model.pkl

Run this first before starting app.py.

Usage:
  # Make sure MLflow server is running first:
  #   mlflow server --host 127.0.0.1 --port 5000
  #
  python train.py
"""

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change these values and re-run to experiment!
# Each run is logged separately in MLflow so you can compare them.
# ──────────────────────────────────────────────────────────────────────────────
PARAMS = {
    "max_features": 5000,    # how many unique words to track
    "ngram_range":  (1, 2),  # (1,1)=single words, (1,2)=words+pairs
    "C":            1.0,     # regularization (lower = simpler model)
    "solver":       "liblinear",
    "test_size":    0.2,     # 20% of data used for testing
    "random_state": 42,      # for reproducible splits
}

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Load the dataset
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  MLOps Demo — Spam Detection Training Script")
print("=" * 60)

print("\n[1/5] Loading data from emails.csv...")
df = pd.read_csv("emails.csv", on_bad_lines="skip")
print(f"      Loaded {len(df)} emails total")
print(f"      Class distribution:")
print(df["label"].value_counts().to_string(header=False))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Prepare features (X) and labels (y)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Preparing features and labels...")
X = df["text"].values    # array of email text strings
y = df["label"].values   # array of "spam" or "ham" strings

# Split into training set (to learn from) and test set (to evaluate on)
# IMPORTANT: the model never sees test data during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=PARAMS["test_size"],
    random_state=PARAMS["random_state"],
    stratify=y,  # keep same spam/ham ratio in both splits
)
print(f"      Training set: {len(X_train)} emails")
print(f"      Test set:     {len(X_test)} emails")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Connect to MLflow and start an experiment run
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Connecting to MLflow at http://127.0.0.1:5000...")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("spam-detection")

print("      Starting MLflow run...")

with mlflow.start_run(run_name="tfidf_logreg_v1"):

    # Log all our parameters so this run is reproducible
    mlflow.log_params({
        "max_features": PARAMS["max_features"],
        "ngram_range":  str(PARAMS["ngram_range"]),
        "C":            PARAMS["C"],
        "solver":       PARAMS["solver"],
        "test_size":    PARAMS["test_size"],
        "dataset_size": len(df),
    })

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: Build and train the ML pipeline
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[4/5] Building and training the pipeline...")
    print("      TfidfVectorizer → converts text to number arrays")
    print("      LogisticRegression → learns spam vs ham patterns")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=PARAMS["max_features"],
                ngram_range=PARAMS["ngram_range"],
                stop_words="english",      # ignore common words like "the", "is"
                lowercase=True,
            ),
        ),
        (
            "model",
            LogisticRegression(
                C=PARAMS["C"],
                solver=PARAMS["solver"],
                max_iter=1000,
            ),
        ),
    ])

    # .fit() trains the pipeline:
    #   First TfidfVectorizer learns vocabulary from X_train
    #   Then LogisticRegression learns spam patterns
    pipeline.fit(X_train, y_train)
    print("      Training complete!")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5: Evaluate and log metrics
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")

    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, pos_label="spam")
    precision = precision_score(y_test, y_pred, pos_label="spam")
    recall    = recall_score(y_test, y_pred, pos_label="spam")

    # Log all metrics to MLflow
    mlflow.log_metrics({
        "accuracy":       round(accuracy,  4),
        "f1_score":       round(f1,        4),
        "precision":      round(precision, 4),
        "recall":         round(recall,    4),
    })

    print(f"\n{'─'*40}")
    print(f"  RESULTS:")
    print(f"  Accuracy : {accuracy:.1%}")
    print(f"  F1 Score : {f1:.3f}")
    print(f"  Precision: {precision:.3f}  ← of emails we called spam, how many were really spam?")
    print(f"  Recall   : {recall:.3f}  ← of all real spam, how many did we catch?")
    print(f"{'─'*40}")
    print()
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Save the full pipeline to MLflow artifact storage
    mlflow.sklearn.log_model(pipeline, "spam_pipeline")

    # Also save locally as spam_model.pkl (used by app.py)
    joblib.dump(pipeline, "spam_model.pkl")

    print("  Model saved to: spam_model.pkl")
    print("  Run logged to:  MLflow (http://127.0.0.1:5000)")
    print()
    print("=" * 60)
    print("  NEXT STEP: python -m uvicorn app:app --reload")
    print("             Then open http://127.0.0.1:8000/docs")
    print("=" * 60)
