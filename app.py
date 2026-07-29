import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Ensure Flask locates the template folder correctly relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "ada_model.pkl")

# Load model safely
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"Template Error: {str(e)}", 500


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model pickle file not loaded properly."}), 500

    try:
        if request.is_json:
            data = request.get_json()
            features = data.get("features", [])
        else:
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

        input_array = np.array(features).reshape(1, -1)
        prediction = int(model.predict(input_array)[0])

        response = {"prediction": prediction}

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_array)[0].tolist()
            response["probabilities"] = probabilities

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
