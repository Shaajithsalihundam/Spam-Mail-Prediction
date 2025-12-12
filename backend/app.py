import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")

# Load model + vectorizer
VECT_PATH = os.path.join(ARTIFACT_DIR, "tfidf.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "spam_model.joblib")

feature_extraction = joblib.load(VECT_PATH)
model = joblib.load(MODEL_PATH)

app = FastAPI()

# Mount static folder (CSS, HTML, etc.)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Home route → serve index.html
@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Prediction input model
class TextIn(BaseModel):
    text: str


# Prediction route
@app.post("/predict")
def predict_text(payload: TextIn):
    text = payload.text

    X_input = feature_extraction.transform([text])
    pred = model.predict(X_input)[0]
    prob = float(np.max(model.predict_proba(X_input)))

    label = "spam" if pred == 1 else "ham"

    return {
        "prediction": label,
        "probability": prob
    }
