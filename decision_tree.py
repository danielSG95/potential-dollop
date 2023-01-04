import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from io import StringIO 
def decision_tree(data):
    print('just a test ')
    st.header('Arbol de decision')
    col1, col2 = st.columns(2)
    with col1:
        features = st.multiselect('Selecciona las features', data.columns.values.tolist())
    with col2:
        target = st.selectbox('Selecionna el target', data.columns)

    X = data[features]
    y = data[target]

    lab = preprocessing.LabelEncoder()
    y = lab.fit_transform(y)

    for i in X:
        if data[i].dtype == 'object':
            X[i] = lab.fit_transform(X[i])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    #clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf = DecisionTreeClassifier(max_depth=5)

    clf = clf.fit(X_train, y_train)


    y_pred = clf.predict(X_test)
    # evaluate the model
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    fig, ax = plt.subplots()
    plot_tree(clf, filled=True, fontsize=10, proportion=True)
    plt.figure(figsize=[50, 30], dpi=300)
    st.pyplot(fig)

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('arbol.png')
    Image(graph.create_png())

    st.download_button('Descargar imagen', graph.create_png(), file_name='arbol.png')
