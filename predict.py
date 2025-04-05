from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the model
model = joblib.load('titanic_model.pkl')

# Exact columns used during training
training_columns = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Age_cleaned',
    'Embarked_Q', 'Embarked_S',
    'Embarked_cleaned_Q', 'Embarked_cleaned_S'
]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load input as DataFrame
        input_json = request.get_json(force=True)
        input_df = pd.DataFrame([input_json])

        # One-hot encode categorical features
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Reindex to match training columns
        input_df = input_df.reindex(columns=training_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
