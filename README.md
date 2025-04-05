Miranda Secaur
theta-webbing-448023-s7

This project implements an end-to-end MLOps pipeline using the Titanic dataset. It includes model training, containerized inference via Flask + Docker, and full deployment to Google Cloud Run.

## Project Overview

 Data Collection
 Preprocessing
 Model Training
 Model Saving
 API Development
 Containerization
 Cloud Deployment
 Explainable AI
 Dashboard (Looker Studio)

## Model

- Type: `RandomForestClassifier`
- Target: `Sex` (binary classification)
- Training Columns (after one-hot encoding):
    ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Age_cleaned', 'Embarked_Q', 'Embarked_S', 'Embarked_cleaned_Q', 'Embarked_cleaned_S']

live API is hosted at:  
[`https://titanic-api-938821466791.us-central1.run.app`](https://titanic-api-938821466791.us-central1.run.app)

## Sample Request:

```bash
curl -X POST https://titanic-api-938821466791.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Pclass": 3,
        "Age": 28.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Age_cleaned": 28.0,
        "Embarked": "S",
        "Embarked_cleaned": "S"
      }'