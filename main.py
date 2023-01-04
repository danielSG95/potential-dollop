import streamlit as st
import pandas as pd


# importing algorithms files
from linear_regression import linear_regression
from polynomial_regression import polynomial_regression
from red_neuronal import red_neuronal
from gauss import gauss
from arbol_decision import arbol_decision
from decision_tree import decision_tree

file_types = {
    "csv": "text/csv",
    "json": "application/json"
}

operations = {
    "Elige una opcion": 0,
    "Regresion Lineal": 1,
    "Regresion Polinomial": 2,
    "Clasificador Gaussiano": 3,
    "Clasificador de arboles de decision": 4,
    "Redes Neuronales": 5
}

dataset = None
operation_type = ''

file = st.file_uploader("Cargar Archivo", type=['csv', 'xml', 'json'], accept_multiple_files=False, label_visibility="visible")
if file is not None:
    try:

        if file.type == file_types['csv']:
            dataset = pd.read_csv(file)
            operation_type = 1
    except Exception as e:
        print(e)


print(operation_type)


if dataset is not None:
    with st.expander('Tabla de datos'):
        st.dataframe(dataset, height=600, width=900)

    operation_type = st.selectbox('Elige un algoritmo', options=operations, index=0)
    if operation_type == 'Regresion Lineal':
        linear_regression(dataset)
    elif operation_type == 'Regresion Polinomial':
        polynomial_regression(dataset)
    elif operation_type == 'Redes Neuronales':
        red_neuronal(dataset)
    elif operation_type == 'Clasificador Gaussiano':
        gauss(dataset)
    elif operation_type == 'Clasificador de arboles de decision':
        decision_tree(dataset)
