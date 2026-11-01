
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("loan_approval_model.pkl")

FEATURES = [
    "credit_score",
    "loan_percent_income",
    "person_income",
    "loan_amnt"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "loan_approval_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


# Root endpoint
@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "loan-approval-prediction",
        "message": "Loan Approval Prediction API is running"
    }), 200


# Dedicated health check endpoint
@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "loan-approval-prediction"
    }), 200


# Loan prediction endpoint
@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{
        feature: data[feature]
        for feature in FEATURES
    }])

    model = load_model()

    prediction_code = int(model.predict(sample)[0])

    prediction = (
        "APPROVED" if prediction_code == 1 else "REJECTED"
    )

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
