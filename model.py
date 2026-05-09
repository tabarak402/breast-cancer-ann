import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping

def build_and_train_model(X_train, y_train, X_test, y_test):
    
    # 1. MODELİ KUR
    # Rapordaki yapı: Giriş(30) → Gizli(32) → Gizli(16) → Çıkış(1)
    model = keras.Sequential([
        # Giriş katmanı + ilk gizli katman
        # 32 nöron, tanh aktivasyon (rapordaki tansig = tanh)
        keras.layers.Dense(32, activation="tanh", input_shape=(X_train.shape[1],)),
        
        # İkinci gizli katman
        # 16 nöron, sigmoid aktivasyon (rapordaki logsig = sigmoid)
        keras.layers.Dense(16, activation="sigmoid"),
        
        # Çıkış katmanı
        # 1 nöron, sigmoid — 0 ile 1 arası olasılık verir
        # 0.5'ten büyükse Malign, küçükse Benign
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    # 2. MODELİ DERLE
    # optimizer: Adam (raporda belirtilmişti)
    # loss: binary_crossentropy (rapordaki crossentropy)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.03),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,          # 20 epoch boyunca gelişme yoksa dur
        restore_best_weights=True
    )
    
    # Model özetini yazdır
    model.summary()
    
    # 3. MODELİ EĞİT
    # epochs=500 (rapordaki gibi)
    # validation_split=0.15 — eğitim verisinin %15'i doğrulama için ayrılır
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.15,
        verbose=1
    )
    
    # 4. EĞİTİM GRAFİĞİNİ KAYDET
    os.makedirs("outputs", exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"],    label="Eğitim")
    plt.plot(history.history["val_accuracy"], label="Doğrulama")
    plt.title("Model Doğruluğu (Epoch'a Göre)")
    plt.xlabel("Epoch")
    plt.ylabel("Doğruluk")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"],    label="Eğitim")
    plt.plot(history.history["val_loss"], label="Doğrulama")
    plt.title("Model Kaybı (Epoch'a Göre)")
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("outputs/1_training_history.png", dpi=150)
    plt.close()
    print("✅ 1_training_history.png kaydedildi")
    
    return model