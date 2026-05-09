import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_prepare_data(filepath="data/data.csv"):
    # 1. Veriyi oku
    df = pd.read_csv(filepath)
    
    # 2. Gereksiz sütunları at
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    
    # 3. Hedef değişkeni sayıya çevir
    # M (Malignant/Kötü huylu) = 1
    # B (Benign/İyi huylu) = 0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    
    # 4. Özellikler ve hedef ayır
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    
    # 5. Eğitim / Test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)
    
    # 6. Normalize et
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    # 7. Scaler'ı kaydet — API'de kullanmak için
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(scaler, "outputs/scaler.pkl")
    print("✅ Scaler kaydedildi: outputs/scaler.pkl")
    
    print(f"✅ Veri hazır — Eğitim: {X_train.shape[0]} örnek, Test: {X_test.shape[0]} örnek")
    return X_train, X_test, y_train, y_test, scaler