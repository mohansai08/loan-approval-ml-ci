
import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# --------------------------------------------------
# 1. APPLICATION AND LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("loan-approval-api")
app = Flask(__name__)

MODEL_PATH = Path("loan_approval_model.pkl")

FEATURES = [
    "credit_score",
    "loan_percent_income",
    "person_income",
    "loan_amnt"
]

# --------------------------------------------------
# 2. PROMETHEUS METRICS
# --------------------------------------------------

HTTP_REQUESTS = Counter(
    "loan_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "loan_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "loan_api_predictions_total",
    "Total loan prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "loan_api_model_loaded",
    "Whether the loan approval model is loaded"
)

# --------------------------------------------------
# 3. MODEL LOADING
# --------------------------------------------------

def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "loan_approval_model.pkl was not found."
            )

        loaded_model = joblib.load(MODEL_PATH)

        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name
        )

        return loaded_model

    except Exception:
        MODEL_LOADED.set(0)

        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH.name
        )
        raise


try:
    model = load_model()
except Exception:
    logger.exception("event=application_startup_failed")
    raise

logger.info("event=application_started service=loan-approval-prediction")

# --------------------------------------------------
# 4. REQUEST LOGGING AND METRICS MIDDLEWARE
# --------------------------------------------------

@app.before_request
def before_request():
    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )

    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):

    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())

    duration = time.perf_counter() - start_time
    duration_ms = duration * 1000

    # Do not count Prometheus scraping as normal API traffic.
    if request.path != "/metrics":

        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration)

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    response.headers["X-Request-ID"] = request_id

    return response


# --------------------------------------------------
# 5. ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "loan-approval-prediction",
        "message": "Loan Approval Prediction API is running"
    }), 200


# --------------------------------------------------
# 6. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "loan-approval-prediction"
    }), 200


# --------------------------------------------------
# 7. LOAN PREDICTION
# --------------------------------------------------

@app.post("/predict") 
def predict():

    try:
        data = request.get_json(silent=True)

        if not data or not isinstance(data, dict):
            logger.warning(
                "event=invalid_json request_id=%s",
                g.request_id
            )

            return jsonify({
                "error": "JSON request body is required"
            }), 400

        missing_fields = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_fields:
            logger.warning(
                "event=missing_fields request_id=%s fields=%s",
                g.request_id,
                missing_fields
            )

            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        logger.info(
            "event=prediction_request request_id=%s feature_names=%s",
            g.request_id,
            sorted(FEATURES)
        )

        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "APPROVED" if prediction_code == 1 else "REJECTED"
        )

        PREDICTIONS.labels(result="success").inc()

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            g.request_id,
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except Exception:
        PREDICTIONS.labels(result="error").inc()

        logger.exception(
            "event=prediction_failed request_id=%s",
            getattr(g, "request_id", "unknown")
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# --------------------------------------------------
# 8. PROMETHEUS METRICS ENDPOINT
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    return app.response_class(
        response=generate_latest(),
        status=200,
        mimetype=CONTENT_TYPE_LATEST
    )


# --------------------------------------------------
# 9. UNKNOWN ROUTE HANDLING
# --------------------------------------------------

@app.errorhandler(404)
def not_found(error):
    logger.warning(
        "event=route_not_found request_id=%s path=%s",
        getattr(g, "request_id", "unknown"),
        request.path
    )

    return jsonify({
        "error": "Route not found"
    }), 404


# --------------------------------------------------
# 10. APPLICATION STARTUP
# --------------------------------------------------

if __name__ == "__main__":
    logger.info("event=server_starting port=5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
