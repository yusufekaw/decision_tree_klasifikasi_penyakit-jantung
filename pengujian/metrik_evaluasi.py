from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.multiclass import unique_labels

def laporanKlasifikasi(y_test, y_pred):
    laporan = classification_report(y_test, y_pred, zero_division=1)
    return laporan

def CM(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    return cm, TN, FP, FN, TP

def visualisasiCM(cm, y_test, y_pred):
    # Definisikan kelas unik dalam data
    kelas = unique_labels(y_test, y_pred)
    # Tampilkan confusion matrix dalam bentuk plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=kelas, yticklabels=kelas)
    plt.xlabel('Aktual')
    plt.ylabel('Prediksi')
    plt.title('Confusion Matrix')
    plt.show()    