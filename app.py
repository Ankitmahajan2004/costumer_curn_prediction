import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained AdaBoost model
MODEL_PATH = "ada_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Single-file HTML template with inline styling, animations, and Chart.js integration
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Predictive Dashboard</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --accent-purple: #8b5cf6;
            --accent-pink: #ec4899;
            --accent-cyan: #06b6d4;
            --text-light: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-light);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 1100px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            animation: fadeIn 0.8s ease-out;
        }

        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 25px 50px rgba(139, 92, 246, 0.15);
        }

        h1, h2 {
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(to right, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 1.8rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input, select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: var(--text-light);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.3s, box-shadow 0.3s;
        }

        input:focus, select:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 12px rgba(139, 92, 246, 0.4);
        }

        button {
            margin-top: 1.5rem;
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px rgba(236, 72, 153, 0.3);
        }

        button:hover {
            opacity: 0.95;
            transform: scale(1.01);
            box-shadow: 0 12px 25px rgba(236, 72, 153, 0.5);
        }

        .results-section {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: center;
            text-align: center;
        }

        .result-badge {
            margin-top: 1rem;
            padding: 0.6rem 1.5rem;
            border-radius: 30px;
            font-weight: 700;
            font-size: 1.2rem;
            display: inline-block;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--glass-border);
            animation: pulse 2s infinite;
        }

        .chart-wrapper {
            position: relative;
            width: 100%;
            height: 280px;
            margin-top: 1.5rem;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(139, 92, 246, 0); }
            100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }
        }
    </style>
</head>
<body>

<div class="container">
    <!-- Input Form Card -->
    <div class="card">
        <h1>AdaBoost Analytics</h1>
        <p class="subtitle">Enter customer metrics below to generate predictions and dynamic visual insights.</p>
        
        <form id="predictionForm">
            <div class="form-grid">
                <div class="input-group">
                    <label>Age</label>
                    <input type="number" name="Age" value="30" required>
                </div>
                <div class="input-group">
                    <label>Gender</label>
                    <select name="Gender">
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Tenure (Months)</label>
                    <input type="number" name="Tenure" value="12" required>
                </div>
                <div class="input-group">
                    <label>Usage Frequency</label>
                    <input type="number" name="Usage Frequency" value="15" required>
                </div>
                <div class="input-group">
                    <label>Support Calls</label>
                    <input type="number" name="Support Calls" value="2" required>
                </div>
                <div class="input-group">
                    <label>Payment Delay</label>
                    <input type="number" name="Payment Delay" value="1" required>
                </div>
                <div class="input-group">
                    <label>Subscription Type</label>
                    <select name="Subscription Type">
                        <option value="0">Basic</option>
                        <option value="1">Standard</option>
                        <option value="2">Premium</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Contract Length</label>
                    <select name="Contract Length">
                        <option value="0">Monthly</option>
                        <option value="1">Quarterly</option>
                        <option value="2">Annual</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Total Spend ($)</label>
                    <input type="number" step="0.01" name="Total Spend" value="500" required>
                </div>
                <div class="input-group">
                    <label>Last Interaction</label>
                    <input type="number" name="Last Interaction" value="5" required>
                </div>
            </div>
            <button type="submit">Predict & Analyze</button>
        </form>
    </div>

    <!-- Output & Visualization Card -->
    <div class="card results-section">
        <h2>Prediction Analysis</h2>
        <p class="subtitle">Probability Distribution & Prediction Class</p>
        
        <div id="resultContainer">
            <div class="result-badge" id="predictionText">Submit parameters to analyze</div>
        </div>

        <div class="chart-wrapper">
            <canvas id="probaChart"></canvas>
        </div>
    </div>
</div>

<script>
    let probaChart;

    // Initialize Empty Chart
    const ctx = document.getElementById('probaChart').getContext('2d');
    probaChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Class 0 (Retain)', 'Class 1 (Churn)'],
            datasets: [{
                data: [50, 50],
                backgroundColor: ['#06b6d4', '#ec4899'],
                borderColor: 'transparent',
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } },
                    position: 'bottom'
                }
            },
            cutout: '70%'
        }
    });

    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const data = {};
        formData.forEach((value, key) => data[key] = parseFloat(value));

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (result.error) {
                document.getElementById('predictionText').innerText = "Error processing request";
                return;
            }

            // Update Prediction Text
            const predClass = result.prediction;
            document.getElementById('predictionText').innerText = `Predicted Class: ${predClass}`;
            document.getElementById('predictionText').style.color = predClass === 1 ? '#ec4899' : '#06b6d4';

            // Animated Chart Update
            probaChart.data.datasets[0].data = result.probabilities;
            probaChart.update();

        } catch (err) {
            console.error(err);
        }
    });
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file missing or failed to load."}), 500

    try:
        data = request.get_json()
        
        # Explicit feature key ordering matching trained AdaBoost feature set:
        # ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
        #  'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction']
        features = [
            data.get("Age", 0),
            data.get("Gender", 0),
            data.get("Tenure", 0),
            data.get("Usage Frequency", 0),
            data.get("Support Calls", 0),
            data.get("Payment Delay", 0),
            data.get("Subscription Type", 0),
            data.get("Contract Length", 0),
            data.get("Total Spend", 0),
            data.get("Last Interaction", 0)
        ]

        # Reshape array for prediction
        input_array = np.array([features])
        
        prediction = int(model.predict(input_array)[0])
        
        # Calculate probabilities if model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_array)[0].tolist()
        else:
            probabilities = [1.0 if prediction == 0 else 0.0, 1.0 if prediction == 1 else 0.0]

        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
