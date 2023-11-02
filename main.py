from data.pemrosesan_data import ambilData, splitData
from algoritma.decision_tree import Klasifikasi, informasiNode, plotTree, hasilKlasifikasi
#from pengujian.metrik_evaluasi import CM, visualisasiCM, laporanKlasifikasi
from pengujian.manual_metrik import CM, visualisasiCM, Accuracy, Precision, Recall, F1_Score

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

    print("y test")
    print(y_test)

    print("y pred")
    print(y_pred)

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
          ) #menampilkan confusion matrixs
    
    visualisasiCM(cm, y_test, y_pred) #visualisasi confusion matrix

    accuracy = Accuracy(TP, TN, FP, FN)
    precision = Precision(TP, FP)
    recall = Recall(TP, FN)
    f1_score = F1_Score(precision, recall)
    
    print("Accuracy\t: ",round(accuracy,2))
    print("Precision\t: ",round(precision,2))
    print("Recall\t\t: ",round(recall,2))
    print("F1_Score\t: ",round(f1_score,2))
    
    #laporan = laporanKlasifikasi(y_test, y_pred)
    #print("Laporan Klasifikasi:")
    #print(laporan) #menampilkan metrik evaluasi algoritma


    
