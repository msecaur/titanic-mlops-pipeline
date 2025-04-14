import requests
import json

# Prompt for input
pclass = input("Enter Class (1, 2, or 3): ")
age = input("Enter Age: ")
fare = input("Enter Fare: ")

# Build request payload
payload = {
    "Pclass": int(pclass),
    "Age_cleaned": float(age),
    "Fare": float(fare)
}

# Define endpoint
url = "https://titanic-api-938821466791.us-central1.run.app/predict"

# Send POST request
response = requests.post(url, json=payload)

# Print the result
if response.ok:
    result = response.json()
    print("\n Prediction:", result["prediction"])
else:
    print("\n Error:", response.status_code, response.text)
