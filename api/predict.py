from flask import Flask, request, jsonify
import pandas as pd
import joblib
from google.cloud import bigquery
from datetime import datetime
import os

# Load model and features
model = joblib.load("titanic_model.pkl")
features = joblib.load("model_features.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()

    # Ensure input contains all required features
    if not all(key in input_data for key in features):
        return jsonify({"error": f"Missing one of the required features: {features}"}), 400

    # Convert input to DataFrame
    df = pd.DataFrame([input_data], columns=features)

    # Make prediction
    prediction = model.predict(df)[0]

    # Convert prediction to label
    prediction_label = "female" if prediction == 0 else "male"

    # Log prediction to BigQuery
    try:
        client = bigquery.Client()
        table_id = "theta-webbing-448023-s7.titanic_predictions.predictions"

        row = {
            **input_data,
            "prediction": prediction_label,
            "timestamp": datetime.utcnow().isoformat()
        }

        errors = client.insert_rows_json(table_id, [row])
        if errors:
            print("BigQuery logging errors:", errors)
    except Exception as e:
        print("Error logging to BigQuery:", e)

    return jsonify({"prediction": prediction_label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))