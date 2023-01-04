import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def generar_regresion(x, y, x_var, y_var, preddiction, regression):
    figure, auxiliar = plt.subplots()
    auxiliar.scatter(x, y, color='black')
    auxiliar.plot(x, preddiction, color='red')
    plt.title('Coeficiente de regression ' + str(regression))
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid()
    st.pyplot(figure)


def generar_grafica_puntos(x, y, x_var, y_var):
    fig, aux = plt.subplots()
    aux.scatter(x, y, color='red')
    plt.title('Grafica de puntos')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    st.pyplot(fig)


def linear_regression(data):
    st.title('Regresion Lineal')
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.subheader("Variable independiente")
        x_var = st.selectbox('Escoga una opcion', data.columns, key="1")
    with col2:
        st.subheader('Variable dependiente')
        y_var = st.selectbox('Escoga una opcion', data.columns, key="2")

    x = np.asarray(data[x_var]).reshape(-1, 1)
    y = data[y_var]

    rgl = linear_model.LinearRegression()
    rgl.fit(x, y)

    preddiction = rgl.predict(x)
    regression = rgl.coef_

    value = st.number_input('Ingrese un valor para predecir', 0)
    result = 0
    if value != 0:
        result = rgl.predict([[int(value)]])

    texto = str(round(rgl.coef_[0],4))+"X+"+str(round(rgl.intercept_, 4))
    data = {'Nombre': ['prediccion', 'coeficiente de regresion', 'coeficiente det','error', 'ecuacion'],
            'Valor': [result, regression, [r2_score(y, preddiction)], [mean_squared_error(y, preddiction)], texto]}
    st.dataframe(pd.DataFrame(data))
    if st.button('Generar Graficas'):
        col3, col4 = st.columns(2)
        with col3:
            generar_regresion(x, y, x_var, y_var, preddiction, regression)
        with col4:
            generar_grafica_puntos(x, y, x_var, y_var)
