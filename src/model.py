import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import os

def build_and_train_model(X_train, y_train, X_test, y_test):
    
    # 1. MODELİ KUR
    model = keras.Sequential([
        keras.layers.Dense(32, activation="tanh", input_shape=(X_train.shape[1],)),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    # 2. MODELİ DERLE
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.03),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # 3. MODELİ EĞİT
    # Early Stopping: val_loss 20 epoch boyunca iyileşmezse dur
    # restore_best_weights: en iyi ağırlıkları geri yükle
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 4. EĞİTİM GRAFİĞİNİ KAYDET
    os.makedirs("outputs", exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"],     label="Eğitim")
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