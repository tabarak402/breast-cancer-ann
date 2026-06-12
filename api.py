from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

app = FastAPI(title="Meme Kanseri Tahmin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modeli ve scaler'ı yükle
model = tf.keras.models.load_model("outputs/ann_model.keras")
scaler = joblib.load("outputs/scaler.pkl")

# Giriş verisi formatı — 30 özellik
class PatientData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"mesaj": "Meme Kanseri Tahmin API çalışıyor!"}

@app.post("/predict")
def predict(data: PatientData):
    # Veriyi numpy dizisine çevir
    X = np.array(data.features).reshape(1, -1)
    
    # Normalize et
    X_scaled = scaler.transform(X)
    
    # Tahmin yap
    prob = float(model.predict(X_scaled)[0][0])
    label = "Malignant (Kötü Huylu)" if prob >= 0.5 else "Benign (İyi Huylu)"
    
    return {
        "tahmin": label,
        "olasilik": round(prob, 4),
        "risk": "YÜKSEK" if prob >= 0.5 else "DÜŞÜK"
    }