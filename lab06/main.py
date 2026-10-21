import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Diabetes Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class _FallbackDiabetesModel:
    """Fallback wrapper with the exact trained LogisticRegression weights from Mymdoel.pkl."""
    feature_names_in_ = np.array([
        "Age", "BMI", "BloodPressure", "Insulin", "Glucose", "DiabetesPedigreeFunction"
    ])
    coef_ = np.array([[0.04559683, 0.09394991, -0.00964773, 0.00442035, 0.03010364, 0.48521205]])
    intercept_ = np.array([-9.2358725])

    def predict_proba(self, df):
        x = df[list(self.feature_names_in_)].to_numpy(dtype=float)
        logits = self.intercept_[0] + x @ self.coef_[0]
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, df):
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)


MODEL_PATH = Path(__file__).parent / "Mymdoel.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception:
    model = _FallbackDiabetesModel()


FEATURES = list(model.feature_names_in_)


class PredictRequest(BaseModel):
    Glucose: float
    BloodPressure: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.get("/")
def home():
    return {"message": "Diabetes API is Running. /docs for API documentation."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    df = pd.DataFrame([payload])[FEATURES]
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {
        "predicted_label": pred,
        "prediction": "Diabetic" if pred == 1 else "Not Diabetic",
        "probability": round(prob, 3),
    }

# python -m pip install fastapi uvicorn pandas scikit-learn
# python -m uvicorn main:app --reload --port 8000
