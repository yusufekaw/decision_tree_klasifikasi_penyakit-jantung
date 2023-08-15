import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#fungsi untuk memuat dataset
def ambilData():
    # Path ke file dataset
    #path = "dataset/heart.csv"
    # Membaca dataset menggunakan Pandas
    dataset = pd.read_csv("/home/ucup/projects/python/fajarfr/data/dataset/heart.csv")
    return dataset

def splitData(dataset):
    # Pisahkan fitur dan variabel target
    X = dataset.drop('target', axis=1)
    y = dataset['target']
    # Bagi data menjadi data pelatihan dan data pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test