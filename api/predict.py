from flask import Flask, request, jsonify
import pandas as pd
import joblib
from google.cloud import bigquery
from datetime import datetime
import os

app = Flask(__name__)

# Load model + features
model = joblib.load("titanic_model.pkl")
feature_columns = joblib.load("model_features.pkl")

# Set up BigQuery client
bq_client = bigquery.Client()
table_id = "theta-webbing-448023-s7.titanic_predictions.predictions"  # project.dataset.table

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_json = request.get_json(force=True)

        # Convert to DataFrame and one-hot encode
        input_df = pd.DataFrame([input_json])
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]

        # Add timestamp + prediction to original input
        row = input_json.copy()
        row["timestamp"] = datetime.utcnow().isoformat()
        row["prediction"] = prediction

        # Prepare row as a BigQuery row
        errors = bq_client.insert_rows_json(table_id, [row])
        if errors:
            print("BigQuery insertion errors:", errors)

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
