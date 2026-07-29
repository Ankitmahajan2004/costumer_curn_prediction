import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "ada_model.pkl"

# Load the trained AdaBoost model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in the root directory.")

# Feature mapping for display names
FEATURE_NAMES = [
    "Age", "Gender", "Tenure", "Usage Frequency", 
    "Support Calls", "Payment Delay", "Subscription Type", 
    "Contract Length", "Total Spend", "Last Interaction"
]


@app.route("/", methods=["GET"])
def index():
    """Renders the HTML user interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint for predictions (handles both JSON and Form submissions)."""
    try:
        if request.is_json:
            data = request.get_json()
            features = data.get("features", [])
        else:
            # Handle standard HTML form submission
            features = [
                float(request.form.get("age", 0)),
                float(request.form.get("gender", 0)),
                float(request.form.get("tenure", 0)),
                float(request.form.get("usage_frequency", 0)),
                float(request.form.get("support_calls", 0)),
                float(request.form.get("payment_delay", 0)),
                float(request.form.get("subscription_type", 0)),
                float(request.form.get("contract_length", 0)),
                float(request.form.get("total_spend", 0)),
                float(request.form.get("last_interaction", 0))
            ]

        # Reshape input for single sample prediction
        input_array = np.array(features).reshape(1, -1)

        # Predict
        prediction = int(model.predict(input_array)[0])

        response = {"prediction": prediction}

        # Predict probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_array)[0].tolist()
            response["probabilities"] = probabilities

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
