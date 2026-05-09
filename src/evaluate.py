import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
import os

def evaluate_model(model, X_test, y_test):
    os.makedirs("outputs", exist_ok=True)

    # 1. TAHMİN YAP
    # Model 0-1 arası olasılık verir
    # 0.5'ten büyükse Malign (1), küçükse Benign (0)
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # 2. METRİKLER
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*50)
    print("  MODEL PERFORMANS SONUÇLARI")
    print("="*50)
    print(f"  Doğruluk (Accuracy) : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  AUC-ROC             : {auc:.4f}")
    print("\n" + classification_report(y_test, y_pred,
          target_names=["Benign (İyi Huylu)", "Malignant (Kötü Huylu)"]))

    # 3. CONFUSION MATRIX GRAFİĞİ
    # Hangi vakaları doğru, hangilerini yanlış tahmin etmiş gösterir
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig("outputs/2_confusion_matrix.png", dpi=150)
    plt.close()
    print("✅ 2_confusion_matrix.png kaydedildi")

    # 4. ROC EĞRİSİ
    # Modelin ayırt edici gücünü gösterir
    # AUC 1'e ne kadar yakınsa model o kadar iyi
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#E76F8A", lw=2,
             label=f"ROC Eğrisi (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Rastgele Tahmin")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Eğrisi", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/3_roc_curve.png", dpi=150)
    plt.close()
    print("✅ 3_roc_curve.png kaydedildi")

    # 5. TAHMİN DAĞILIMI GRAFİĞİ
    # Modelin Benign ve Malignant için verdiği olasılıkların dağılımı
    plt.figure(figsize=(7, 4))
    plt.hist(y_prob[y_test == 0], bins=30, alpha=0.6,
             color="#4C72B0", label="Benign (Gerçek)")
    plt.hist(y_prob[y_test == 1], bins=30, alpha=0.6,
             color="#E76F8A", label="Malignant (Gerçek)")
    plt.axvline(0.5, color="black", linestyle="--", label="Karar Sınırı (0.5)")
    plt.xlabel("Tahmin Olasılığı")
    plt.ylabel("Frekans")
    plt.title("Tahmin Olasılığı Dağılımı", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/4_prediction_distribution.png", dpi=150)
    plt.close()
    print("✅ 4_prediction_distribution.png kaydedildi")

    print("\n🎉 Tüm grafikler outputs/ klasörüne kaydedildi!")
    return acc, auc