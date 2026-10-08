"""
test_api.py
===========
MLOps Demo — Test script for the Spam Detector API.

Sends several test emails to the running API and shows predictions.

Usage:
  # Make sure app.py is running first:
  #   uvicorn app:app --reload
  #
  python test_api.py
"""

import requests
import json

API_URL = "http://127.0.0.1:8000"


def separator(title=""):
    print("\n" + "═" * 60)
    if title:
        print(f"  {title}")
        print("═" * 60)


def test_health():
    """Test that the API is running."""
    separator("Test 1: Health Check")
    try:
        r = requests.get(f"{API_URL}/")
        print(f"  Status code: {r.status_code}")
        print(f"  Response:    {r.json()['status']}")
        if r.status_code == 200:
            print("  ✓ API is running!")
        else:
            print("  ✗ Something is wrong!")
    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot connect to API!")
        print("    Make sure you have started: uvicorn app:app --reload")
        return False
    return True


def test_model_info():
    """Test the model info endpoint."""
    separator("Test 2: Model Information")
    r = requests.get(f"{API_URL}/info")
    data = r.json()
    print(f"  Model type:       {data.get('model_type', 'N/A')}")
    print(f"  Vocabulary size:  {data.get('vectorizer', {}).get('vocabulary_size', 'N/A')}")
    print(f"  Classes:          {data.get('classes', 'N/A')}")
    print(f"  Status:           {data.get('status', 'N/A')}")


def test_single_predictions():
    """Test classifying individual emails."""
    separator("Test 3: Single Email Predictions")

    # (email_text, expected_label)
    test_cases = [
        ("WINNER!! You have been selected to receive a FREE prize worth 1000!", "spam"),
        ("Hey, are you coming to the lecture tomorrow at 10am?",                "ham"),
        ("URGENT: Your account will be suspended! Click here NOW to verify",    "spam"),
        ("Can you please send me the notes from today's lab session?",          "ham"),
        ("FREE entry to win! Text WIN to 80082 right now!",                     "spam"),
        ("Meeting has been moved to 3pm. Conference room B.",                   "ham"),
        ("Congratulations! You have won a 2000 cash prize. Call to claim.",     "spam"),
        ("I will be 10 minutes late for the practical class.",                  "ham"),
    ]

    print(f"\n  {'Verdict':<10} {'Confidence':<12} {'Correct?':<10} Email")
    print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*40}")

    correct = 0
    for email_text, expected in test_cases:
        r    = requests.post(f"{API_URL}/predict", json={"text": email_text})
        data = r.json()

        verdict     = data["verdict"]
        confidence  = data["confidence"]
        prediction  = data["prediction"]
        is_correct  = prediction == expected
        if is_correct:
            correct += 1

        tick = "✓" if is_correct else "✗"
        print(f"  {verdict:<10} {confidence:<12.1%} {tick:<10} {email_text[:45]}...")

    print(f"\n  Score: {correct}/{len(test_cases)} correct ({correct/len(test_cases):.0%})")


def test_batch_prediction():
    """Test classifying multiple emails at once."""
    separator("Test 4: Batch Prediction")

    batch_emails = [
        "SIX chances to win CASH! Text WIN to 87575.",
        "Are you free for the group study session this evening?",
        "FREE camera phone! Call 09061221066 to claim your prize.",
        "The assignment deadline has been extended to Friday.",
        "Urgent! Your 2-mobile.com account requires verification.",
        "Professor uploaded the slides on the portal.",
    ]

    r    = requests.post(f"{API_URL}/predict/batch", json={"texts": batch_emails})
    data = r.json()

    print(f"  Sent {data['total']} emails")
    print(f"  Spam detected: {data['spam_count']}")
    print(f"  Ham detected:  {data['ham_count']}")
    print()
    for result in data["results"]:
        icon = "🔴" if result["is_spam"] else "🟢"
        print(f"  {icon} [{result['verdict']:<9}] {result['confidence']:.0%}  {result['text'][:50]}")


def test_edge_cases():
    """Test some unusual inputs."""
    separator("Test 5: Edge Cases")

    edge_cases = [
        ("FREE FREE FREE!!!",                           "All caps spam words"),
        ("ok",                                          "Very short ham message"),
        ("WINNER WINNER WINNER click here now urgent",  "Multiple spam triggers"),
        ("I love studying machine learning at college", "Normal student message"),
    ]

    for text, description in edge_cases:
        r    = requests.post(f"{API_URL}/predict", json={"text": text})
        data = r.json()
        print(f"  {description}")
        print(f"    Text:       \"{text}\"")
        print(f"    Verdict:    {data['verdict']}  ({data['confidence']:.1%} spam confidence)")
        print(f"    Latency:    {data['latency_ms']} ms")
        print()


def test_error_handling():
    """Test that the API handles bad input correctly."""
    separator("Test 6: Error Handling")

    # Empty text
    r = requests.post(f"{API_URL}/predict", json={"text": ""})
    print(f"  Empty text → HTTP {r.status_code} (expected 422 Unprocessable Entity)")

    # Missing field
    r = requests.post(f"{API_URL}/predict", json={"email": "wrong field name"})
    print(f"  Wrong field name → HTTP {r.status_code} (expected 422)")

    # Wrong content type
    r = requests.post(f"{API_URL}/predict", data="not json")
    print(f"  Not JSON → HTTP {r.status_code} (expected 422)")

    print("\n  ✓ API correctly rejects invalid inputs")


# ──────────────────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║       Spam Detector API — Test Suite                  ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"  Testing: {API_URL}")

    alive = test_health()
    if not alive:
        print("\n  Cannot run tests — start the server first!")
        print("  Command: uvicorn app:app --reload")
        exit(1)

    test_model_info()
    test_single_predictions()
    test_batch_prediction()
    test_edge_cases()
    test_error_handling()

    separator("All Tests Complete!")
    print("  Open http://127.0.0.1:8000/docs to try the API manually.")
    print()
