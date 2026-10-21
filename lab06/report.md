# Lab 06 Report - Deploying the Diabetes Prediction Model with FastAPI

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Instructor:** Dr. Muhammad Awais  

## 1. Overview of Lab 06

In this lab, I deployed the Logistic Regression diabetes classification model (`Mymdoel.pkl`) as a web application using **FastAPI** on the backend (`main.py`) and a simple **HTML + JavaScript** form on the frontend (`index.html`).

* **`Mymdoel.pkl`:** Pickled `scikit-learn` `LogisticRegression` model trained on 6 features: `['Age', 'BMI', 'BloodPressure', 'Insulin', 'Glucose', 'DiabetesPedigreeFunction']`.
* **`main.py`:** FastAPI server with CORS enabled (`CORSMiddleware`) so the browser page can call the API from `localhost`. It defines a `PredictRequest` Pydantic schema to validate input numbers and exposes three routes:
  1. `GET /` - Returns `{"message": "Diabetes API is Running. /docs for API documentation."}`
  2. `GET /health` - Returns `{"status": "ok"}`
  3. `POST /predict` - Converts the JSON request into a Pandas DataFrame ordered by `FEATURES = list(model.feature_names_in_)`, runs `model.predict(df)` and `model.predict_proba(df)`, and returns `predicted_label`, `prediction` (`"Diabetic"` or `"Not Diabetic"`), and `probability`.
* **`index.html`:** Frontend page with 6 number input fields (`Glucose`, `BloodPressure`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`) and a **Predict** button that sends a `POST` request to `http://localhost:8000/predict` using `fetch()` and displays the result on the page.

## 2. How to Run the Project Locally

1. Install the required libraries:
   ```bash
   python -m pip install fastapi uvicorn pandas scikit-learn
   ```
2. Start the FastAPI backend server inside the `lab06/` folder:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```
3. Open `http://localhost:8000/docs` in the browser to test the interactive Swagger UI, or open `index.html` in the browser, enter the 6 patient values, and click **Predict**.

## 3. Sample API Test Results

| Test Case | Input Values (`Glucose, BP, Insulin, BMI, DPF, Age`) | Returned JSON Response |
| :--- | :--- | :--- |
| **Patient 1 (Low Risk)** | `112, 75.0, 102.5, 35.7, 0.148, 21` | `{"predicted_label": 0, "prediction": "Not Diabetic", "probability": 0.148}` |
| **Patient 2 (High Risk)** | `168, 84.0, 190.0, 38.2, 0.685, 48` | `{"predicted_label": 1, "prediction": "Diabetic", "probability": 0.877}` |
