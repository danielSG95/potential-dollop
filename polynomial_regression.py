import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def polynomial_regression(data):
    st.title('Regresion Polinomial')
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        x_var = st.selectbox('Variable independiente (x)', data.columns, key="1")
    with col2:
        y_var = st.selectbox('Varialbe dependiente (y)', data.columns, key="2")
    with col3:
        degree = st.number_input('Grado del polinomio', value=2)

    prediccion = st.number_input('Ingrese un valor para predecir', 0)

    x = data[x_var]
    y = data[y_var]

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    polinomial_feature = PolynomialFeatures(degree=degree)
    X_TRANSF = polinomial_feature.fit_transform(x)

    model = LinearRegression()
    model.fit(X_TRANSF, y)
    
    Y_NEW = model.predict(X_TRANSF)

    rmse = np.sqrt(mean_squared_error(y,Y_NEW))
    r2 = r2_score(y, Y_NEW)

    x_new_min = 0.0
    x_new_max = float(prediccion)  # el calculo de la prediccion

    X_NEW = np.linspace(x_new_min, x_new_max, 50)
    X_NEW = X_NEW[:, np.newaxis]

    X_NEW_TRANSF = polinomial_feature.fit_transform(X_NEW)
    Y_NEW = model.predict(X_NEW_TRANSF)

    coeficientes = model.coef_
    coeficientes = np.asarray(coeficientes)

    auxcoeficientes = coeficientes.reshape(-1, 1)
    intercept = model.intercept_
    intercept = str(model.intercept_).replace('[', '')
    intercept = str(intercept).replace(']', '')

    equation = get_equation(degree, auxcoeficientes, intercept)
    data = {'Nombre': ['RMSE', 'R2', 'Prediccion', 'Ecuacion'],
            'Value': [rmse, r2, Y_NEW[Y_NEW.size-1], equation]}

    st.dataframe(pd.DataFrame(data=data))

    if st.button('Generar Graficas'):
        col3, col4 = st.columns(2)
        with col3:
            generar_regresion(X_NEW, Y_NEW, x_new_min, x_new_max, x_var, y_var)
        with col4:
            generar_puntos(x, y, x_var, y_var)


def generar_regresion(X_NEW, Y_NEW, x_new_min, x_new_max, x_var, y_var):
    fig, ax = plt.subplots()
    plt.plot(X_NEW, Y_NEW, color='red', linewidth=3)
    ax.scatter(X_NEW, Y_NEW, color='black')
    plt.grid()
    plt.xlim(x_new_min, x_new_max)
    plt.title("Regresion polinomial", fontsize=10)
    plt.xlabel(x_var)
    plt.ylabel(y_var)

    st.pyplot(fig)


def generar_puntos(x, y, x_var, y_var):
    fig2, ax2 = plt.subplots()
    plt.grid()
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title("Grafica de puntos", fontsize=10)
    ax2.scatter(x, y, color='black')
    st.pyplot(fig2)


def get_equation(grado, auxcoeficientes, intercept):
    ecuacion = ''
    for i in range(int(grado), 0, -1):
        aux = str(auxcoeficientes[i]).replace('[', '')
        aux = aux.replace(']', '')
        ecuacion += str(aux)+'X^'+str(i)+' + '
    ecuacion += str(intercept)

    return ecuacion
