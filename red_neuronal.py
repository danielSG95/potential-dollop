import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
def red_neuronal(data):
    st.title('Red Neuronal')
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox('Seleccione la columna x', data.columns)
    with col2:
        y_var = st.selectbox('Seleccione la columna y', data.columns)

    predict = st.number_input('Ingrese el valor a predecir')
    predict = [[int(predict)]]

    # x = np.asarray(data[x_var]).reshape(-1, 1)
    x = data[x_var]
    y = data[y_var]

    lab = preprocessing.LabelEncoder()
    if data[x_var].dtype == 'object':
        x = lab.fit_transform(x)

    X = x[:, np.newaxis]
    i = 0
    fig, ax = plt.subplots()
    plt.rc('font', size = 10)
    colors = ['teal', 'pink', 'brown', 'hotpink', 'orchid', 'aqua', 'green', 'blue', 'yellow', 'purple', 'black',
              'tomato', 'salmon', 'olive', 'chocolate', 'wheat']
    while True:
        i = i + 1
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        mlr = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)
        mlr.fit(X_train, y_train)

        plt.scatter(X_train, y_train, color='orange', label="Entrenamiento" if i == 1 else "test")
        plt.scatter(X_test, y_test, color='green', label="Test" if i == 1 else "")
        plt.scatter(X, mlr.predict(X), c=np.random.rand(3,), label="Iteracion " + str(i))
        plt.legend(["Entrenamiento", "Test", "Iteracion: " + str(i)])
        if mlr.score(X_train, y_train) > 0.8 or i == 20:
            break

    st.subheader('Resultado de la prediccion')
    st.write(mlr.predict(predict))

    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.figure(figsize=(20, 10))
    st.pyplot(fig)