from data.pemrosesan_data import ambilData, splitData
from algoritma.decision_tree import Klasifikasi, informasiNode, plotTree, hasilKlasifikasi
from pengujian.metrik_evaluasi import CM, visualisasiCM, laporanKlasifikasi

if __name__=="__main__":

    dataset = ambilData() # Load dataset

    print("Hasil Import Dataset") 
    print(dataset) # menampilkan dataset

    #Split dataset menjadi data training dan testing
    X, y, X_train, X_test, y_train, y_test = splitData(dataset)

    print("Data training:")
    print(X_train) #menampilkan data training

    print("Data testing:")
    print(X_test) #menampilkn data testing

    model, y_pred = Klasifikasi(X_train, y_train, X_test)#proses klasifikasi

    info_node = informasiNode(model, X.columns) 
    print("informasi node:")
    print(info_node)#informasi node

    plotTree(model, X) #ploting decision tree

    hasil = hasilKlasifikasi(X_test, y_pred)
    print("hasil klasifikasi;")
    print(hasil) #hasil klasifikasi

    cm, TN, FP, FN, TP = CM(y_test, y_pred)
    print("confusion matrix:")
    print(
            "TP :",TP,"\n"
            "FP :",FP,"\n"
            "FN :",FN,"\n"
            "TN :",TN,"\n"
          ) #menampilkan confusion matrix
    
    visualisasiCM(cm, y_test, y_pred) #visualisasi confusion matrix

    laporan = laporanKlasifikasi(y_test, y_pred)
    print("Laporan Klasifikasi:")
    print(laporan) #menampilkan metrik evaluasi algoritma


    
