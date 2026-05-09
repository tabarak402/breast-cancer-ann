from src.preprocess import load_and_prepare_data
from src.model import build_and_train_model
from src.evaluate import evaluate_model
from tensorflow.keras.callbacks import EarlyStopping

print(" Meme Kanseri ANN Projesi Başlıyor...\n")

# 1. Veriyi hazırla
print(" Adım 1: Veri yükleniyor ve hazırlanıyor...")
X_train, X_test, y_train, y_test, scaler = load_and_prepare_data("data/data.csv")

# 2. Modeli kur ve eğit
print("\n Adım 2: Model kuruluyor ve eğitiliyor...")
model = build_and_train_model(X_train, y_train, X_test, y_test)

# 3. Değerlendir
print("\n Adım 3: Model değerlendiriliyor...")
acc, auc = evaluate_model(model, X_test, y_test)

# 4. Modeli kaydet
model.save("outputs/ann_model.keras")
print(f"\n Model kaydedildi: outputs/ann_model.keras")
print(f"\n Sonuç — Doğruluk: {acc*100:.2f}% | AUC: {auc:.4f}")


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