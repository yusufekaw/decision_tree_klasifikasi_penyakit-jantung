# Buat model Decision Tree
    model = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=15, random_state=0)

    # Latih model menggunakan data pelatihan
    model.fit(X_train, y_train)

    # Prediksi data pengujian
    y_pred = model.predict(X_test)
    kelas = pd.Series(y_pred, name='Predicted', index=X_test.index)
    print(kelas)

    print(y_pred)

    results_df = pd.concat([X_test, kelas], axis=1)
    print(results_df)

    # Evaluasi kinerja model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Akurasi: {accuracy}')

    # Tampilkan classification report tanpa warning
    print(classification_report(y_test, y_pred, zero_division=1))  # Set zero_division to 1

    # Dapatkan informasi Gain dan Entropy untuk setiap node dalam bentuk dataframe
    def get_node_info(tree, feature_names):
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        impurity = tree.tree_.impurity

        data = {
            'Feature': [feature_names[i] if i >= 0 else None for i in feature],
            'Threshold': threshold,
            'Impurity': impurity
        }

        info_node = pd.DataFrame(data)
        return info_node

    node_info = get_node_info(model, feature_names=X.columns)
    print(node_info)

    # Visualisasikan Decision Tree dengan ukuran dan depth yang disesuaikan
    plt.figure(figsize=(10, 5))
    tree.plot_tree(model, feature_names=X.columns, class_names=['No Heart Disease', 'Heart Disease'],
                filled=True, rounded=True, fontsize=6, max_depth=15)  # Adjust max_depth as needed
    
    # Tambahkan legenda
    class_names = ['No Heart Disease', 'Heart Disease']
    colors = ['lightblue', 'lightcoral']
    legend_labels = [f'{class_names[i]} ({i})' for i in range(len(class_names))]

    patches = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i], 
            markerfacecolor=colors[i], markersize=10) for i in range(len(class_names))]

    plt.legend(handles=patches, loc='upper right')

    # Tambahkan penjelasan mengenai elemen-elemen pohon keputusan
    plt.text(0.0, 0.9, "Root Node", transform=plt.gca().transAxes, ha="left", fontsize=6)
    plt.text(0.0, 0.8, "Internal Node", transform=plt.gca().transAxes, ha="left", fontsize=6)
    plt.text(0.0, 0.7, "Leaf Node", transform=plt.gca().transAxes, ha="left", fontsize=6)
    plt.text(0.0, 0.6, "Feature", transform=plt.gca().transAxes, ha="left", fontsize=6)
    plt.text(0.0, 0.5, "Threshold ", transform=plt.gca().transAxes, ha="left", fontsize=6)
    plt.text(0.5, 0.4, "Impurity", transform=plt.gca().transAxes, ha="left", fontsize=6)

    plt.show()