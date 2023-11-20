import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.multiclass import unique_labels

def CM(y_test, y_pred):
    y_test = y_test.tolist()
    TP = 0
    FP = 0 
    TN = 0
    FN = 0

    # Menghitung TP, FP, TN, FN
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_test[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_test[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            FN += 1
    cm = [[TP, TN], [FP, FN]]

    return cm, TP, FP, TN, FN

def visualisasiCM(cm, y_pred, y_test):
    # Definisikan kelas unik dalam data
    kelas = unique_labels(y_pred, y_test)
    # Tampilkan confusion matrix dalam bentuk plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=kelas, yticklabels=kelas)
    plt.xlabel('Aktual')
    plt.ylabel('Prediksi')
    plt.title('Confusion Matrix')
    plt.show()

def Accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + FP + TN + FN)

def Precision(TP, FP):
    return TP / (TP + FP)

def Recall(TP, FN):
    return TP / (TP + FN)

def F1_Score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)