from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from sklearn import preprocessing
import streamlit as st
import numpy as np
from sklearn.naive_bayes import GaussianNB


def gauss(data):
    st.title('Clasificador Gaussiano')
    col1, col2 = st.columns(2)
    with col1:
        param = st.selectbox('Parmetro de aproximacion', data.columns)
    with col2:
        remove_column = st.selectbox('Quitar columna', data.columns)

    listaa = data.columns.values.tolist()

    listaa.remove(param)
    if remove_column != 'Seleccionar':
        listaa.remove(remove_column)

    result = data[param]

    listadedf = []
    for i in listaa:
        aux = data[i]
        aux = np.asarray(aux)
        
        listadedf.append(aux)
    listadedf = np.array(listadedf)

    le = preprocessing.LabelEncoder()

    listafittransform = []
    for x in listadedf:
        listafittransform.append(le.fit_transform(x))

    label = le.fit_transform(result)

    with st.expander("Resultado codificando"):
        featuresencoders = list(zip((listafittransform)))
        featuresencoders = np.array(featuresencoders)
        tamcolumnas = len(listaa)
        tamfilas = featuresencoders.size
        featuresencoders = featuresencoders.reshape(int(tamfilas/tamcolumnas), tamcolumnas)

        # st.dataframe(featuresencoders)

    with st.expander('Matriz de resultados'):
        features = list(zip(np.asarray(listadedf)))
        features = np.asarray(features)
        tamcolumnas = len(features)
        tamfilas = features.size
        features = features.reshape(int(tamfilas/tamcolumnas), tamcolumnas)
        st.dataframe(features)

    model = GaussianNB()
    model2 = GaussianNB()

    model.fit(np.asarray(features), np.asarray(result))
    model2.fit(featuresencoders, label)

    columna = len(listaa)
    texto = "Ingrese "+str(columna)+" valores separados por espacio."
    generate_array = st.button('generar random array')
    x = ''
    if generate_array:
        x = np.random.randint(100, size=(columna))

    predecirresult = st.text_input(texto, x)

    if predecirresult != '':
        entrada = predecirresult.replace('[', '')
        entrada = entrada.replace(']', '')
        entrada = entrada.split(' ')
        map_obj = array_to_int(entrada)
        map_obj = np.array(map_obj)
        predicted = model.predict(np.asarray([map_obj]))

        st.subheader('Resultado de prediccion')
        st.write(predicted)


def array_to_int(array):
    temporal = []
    for i in array:
        try:
            temporal.append(int(i))
        except ValueError:
            continue
    print(temporal)
    return temporal
