from fastapi import FastAPI
import joblib

app = FastAPI(title="Heart Disease Serving")

model = joblib.load("model.pkl")

@app.get("/")
def root():
    return {"status": "model loaded"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}