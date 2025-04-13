# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY titanic_model.pkl .
COPY predict.py .
COPY . .

# Install required packages
RUN pip install flask pandas scikit-learn joblib

# Expose port
EXPOSE 8080

# Run the Flask app
CMD ["python", "predict.py"]