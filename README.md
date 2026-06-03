# Predictive Maintenance ML API

Machine Learning project for predictive maintenance using Logistic Regression, Random Forest and a Neural Network (Keras).

## Features

- Predict machine failures based on sensor data
- Three ML models trained and compared
- FastAPI prediction endpoint with the best performing model
- Classification metrics including Accuracy, Precision, Recall and F1-Score
- Confusion matrix for visual evaluation
- Interactive API documentation with Swagger UI

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Joblib

---

## Project Overview

This project predicts machine failures based on sensor data from industrial machines.

The workflow includes:
- Data preprocessing and feature engineering with Pandas
- Training and comparing three ML models
- Evaluating models with focus on Recall – missing a real failure is more costly than a false alarm
- Saving the best model with Joblib
- Deploying the model as a REST API with FastAPI

---

## Machine Learning Workflow

1. Load and analyze the dataset
2. Clean and preprocess the data
3. Encode categorical features (One-Hot Encoding)
4. Split data into training and test sets
5. Train and evaluate three models: Logistic Regression, Random Forest, Neural Network
6. Save the best model (Random Forest)
7. Deploy prediction API with FastAPI

---

## Model Performance

### Logistic Regression
- Accuracy: 97.4%
- Recall (Failure Detection): 30%

### Random Forest
- Accuracy: 98.45%
- Recall (Failure Detection): 57%

### Neural Network (Keras)
- Accuracy: ~97%
- Recall (Failure Detection): ~31%

Random Forest achieved the best failure detection performance and was selected for deployment.

The dataset is highly imbalanced – only ~3.4% of machines fail. A model that always predicts "No Failure" would reach 96.6% accuracy while detecting zero actual failures. This is why Recall is the critical metric, not Accuracy.

---

## API Usage

Start the API:

```bash
uvicorn api.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST /predict

Example Request:

```json
{
  "air_temperature": 305.0,
  "process_temperature": 315.0,
  "rotational_speed": 2800,
  "torque": 90.0,
  "tool_wear": 250,
  "type_L": false,
  "type_M": true
}
```

Example Response:

```json
{
  "prediction": 1,
  "prediction_label": "Failure"
}
```

---

## Screenshots

### FastAPI Documentation

![FastAPI Docs](screenshots/api_docs.png)

### Prediction Input

![Prediction Input](screenshots/prediction_input.png)

### Prediction Response

![Prediction Response](screenshots/prediction_response.png)

---

## Project Structure

```text
predictive-maintenance-ml-api/
├── api/
│   └── main.py
├── data/
│   └── predictive_maintenance.csv
├── models/
│   └── random_forest_model.pkl
├── notebooks/
│   └── analysis.ipynb
├── screenshots/
├── src/
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn api.main:app --reload
```

Open in browser:

```text
http://127.0.0.1:8000/docs
```