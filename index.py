import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import calendar
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandasai.llm.openai import OpenAI
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import streamlit as st
import pandas as pd
import requests
import openai
from io import StringIO

def cargar_archivo():
    # URL de descarga del archivo CSV en Google Drive
    url = "https://drive.google.com/uc?id=1-8M2DgY9n5UBYeihZd6bQyWE4gsY4yyb"

    # Descargar el archivo CSV
    r = requests.get(url)
    archivo_csv = r.text

    # Convertir los datos en un DataFrame de pandas
    datos = pd.read_csv(StringIO(archivo_csv))
    return datos
 
def cargar_archivoInsta():
    # URL de descarga del archivo CSV en Google Drive
    url = "https://drive.google.com/uc?id=1GWcvFMOABBmg3yp3_SxSxRd6oP16hR77"

    # Descargar el archivo CSV
    r = requests.get(url)
    archivo_csv = r.text

    # Convertir los datos en un DataFrame de pandas
    datos = pd.read_csv(StringIO(archivo_csv))
    return datos

def cargar_archivoMetricas():
    url = "https://drive.google.com/uc?id=1qAxG72MsQhgcUTwWf-1jCT6Aa2QZS0Ht"

    # Descargar el archivo CSV
    r = requests.get(url)
    archivo_csv = r.text

    # Convertir los datos en un DataFrame de pandas
    datos = pd.read_csv(StringIO(archivo_csv))
    return datos

def cargar_archivoSemillero():
    url = "https://drive.google.com/uc?id=1HGMavqdxZO4Nh-jILCxqERHPkcwNT2VW"

    # Descargar el archivo CSV
    r = requests.get(url)
    archivo_csv = r.text

    # Convertir los datos en un DataFrame de pandas
    datos = pd.read_csv(StringIO(archivo_csv))
    return datos

def cargar_archivoProyectos():
    url = "https://drive.google.com/uc?id=1uj3T4P0hKZHldnZqPnlYGltueS2yHpml"

    # Descargar el archivo CSV
    r = requests.get(url)
    archivo_csv = r.text

    # Convertir los datos en un DataFrame de pandas
    datos = pd.read_csv(StringIO(archivo_csv))
    return datos

def cargar_archivoPaises():
    url = "https://drive.google.com/uc?id=1ICeVbqE6YwYCfR_vRdZ0ZYJdXS3oK2Ol"

    # Descargar el archivo CSV
    r = requests.get(url)
    archivo_csv = r.text

    # Convertir los datos en un DataFrame de pandas
    datos = pd.read_csv(StringIO(archivo_csv))
    return datos

def cargar_archivoCiudades():
    url = "https://drive.google.com/uc?id=1xp-XaBlz46cpBG5kwNpS85l-jqOWS7-p"

    # Descargar el archivo CSV
    r = requests.get(url)
    archivo_csv = r.text

    # Convertir los datos en un DataFrame de pandas
    datos = pd.read_csv(StringIO(archivo_csv))
    return datos

def cargar_archivoEdad():
    url = "https://drive.google.com/uc?id=1taxIUst3EPNGPxoHaUkVtGkfrRJOJcp7"

    # Descargar el archivo CSV
    r = requests.get(url)
    archivo_csv = r.text

    # Convertir los datos en un DataFrame de pandas
    datos = pd.read_csv(StringIO(archivo_csv))
    return datos


#Funciones FACEBOOK
def mostrar_analisis(datos):
    datos_numericos = datos.select_dtypes(include=['float64', 'int64'])

    # Columnas en Streamlit
    col1, col2 = st.columns(2)

    #GRAFICO 1
    with col1:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE CORRELACIÓN</strong></h2>", unsafe_allow_html=True)
        correlation_matrix = datos_numericos.corr()
        fig, ax = plt.subplots(figsize=(2, 2))
        plt.rc('font', size=3)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, square=True)
        ax.set_title('Matriz de Correlación')
        plt.gcf().patch.set_linewidth(2)
        plt.gcf().patch.set_edgecolor('black')
        st.pyplot(fig, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de correlación relaciona las variables del conjunto de datos. Es una tabla que muestra la correlación entre cada par de variables. Estos coeficientes van de -1 a 1, indicando la fuerza y dirección de la relación entre las variables. Un valor de 1 implica una correlación positiva perfecta, -1 una correlación negativa perfecta y 0 ausencia de relación.</p>", unsafe_allow_html=True)
    
    #GRAFICO 2
    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE CAJA DE INTERACCIONES Y TIPO DE PUBLICACIÓN</strong></h2>", unsafe_allow_html=True)
        plt.figure(figsize=(3, 1))
        sns.boxplot(x='Tipo de publicación', y='Interacciones', data=datos, palette="deep")
        plt.title('Interacciones por Tipo de Publicación')
        plt.xlabel('Tipo de Publicación')
        plt.ylabel('Interacciones')
        fig2, ax2 = plt.gcf(), plt.gca() 
        st.pyplot(fig2, ax2, use_container_width=False) 
        st.markdown("<p style='text-align: justify;'>El gráfico de caja de interacciones presenta información sobre el número de interacciones según el tipo de publicación, como foto, video, otro y video en vivo. En el eje x se muestra el tipo de publicación, mientras que en el eje y se indica el número de interacciones. </p>", unsafe_allow_html=True)
    

    #GRAFICO 3 
    with col1:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE BARRAS  DE INTERACCIONES POR MESES</strong></h2>", unsafe_allow_html=True)
        interacciones_por_mes = datos.groupby('Fecha_mes')['Interacciones'].sum().reset_index()
        plt.figure(figsize=(3, 1))
        sns.barplot(x='Fecha_mes', y='Interacciones', data=interacciones_por_mes, palette="deep")
        plt.title('Interacciones en Facebook por Mes')
        plt.xlabel('Mes')
        plt.ylabel('Interacciones')
        fig3, ax3 = plt.gcf(), plt.gca()
        st.pyplot(fig3, ax3, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de barras de interacciones presenta información sobre el número de interacciones totales. En el eje x se muestra los meses del conjunto de datos, mientras que en el eje y se indica el número de interacciones totales.</p>", unsafe_allow_html=True)

    #GRAFICO 4
    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE BARRAS TOTAL DEL PERSONAS ALCANZADAS</strong></h2>", unsafe_allow_html=True)
        interacciones_por_mes = datos.groupby('Fecha_mes')['Personas alcanzadas'].sum().reset_index()
        plt.figure(figsize=(3, 1))
        sns.barplot(x='Fecha_mes', y='Personas alcanzadas', data=interacciones_por_mes, palette="deep")
        plt.title('Suma Total de Personas alcanzadas en Facebook por Mes')
        plt.xlabel('Mes')
        plt.ylabel('Suma Total de Personas alcanzadas')
        fig4, ax4 = plt.gcf(), plt.gca()
        st.pyplot(fig4, ax4, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de barras de personas alcanzadas, muestra la totalidad de personas que miraron las publicaciones. En el eje x se muestra los meses del conjunto de datos, mientras que en el eje y se indica el número de personas que miraron las publicaciones.</p>", unsafe_allow_html=True)

    #GRAFICO 5 
    with col1:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE BARRAS INTERACCIONES</strong></h2>", unsafe_allow_html=True)
        interacciones_por_mes = datos.groupby('Tipo de publicación')['Interacciones'].sum().reset_index()
        plt.figure(figsize=(3, 1))
        sns.barplot(x='Tipo de publicación', y='Interacciones', data=interacciones_por_mes, palette="deep")
        plt.title('Suma Total de Interacciones en Facebook por Tipo de publicación')
        plt.xlabel('Tipo de publicación')
        plt.ylabel('Suma Total de Interacciones')
        fig5, ax5 = plt.gcf(), plt.gca()
        st.pyplot(fig5, ax5, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de barras de interacciones, muestra la totalidad de interacciones por tipo de publicación. En el eje x se muestra el tipo de publicación, mientras que en el eje y se indica el número de interacciones totales que se tiene por publicacion.</p>", unsafe_allow_html=True)

    #Grafico 6
    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE BARRAS TIPO DE INTERACCIÓN</strong></h2>", unsafe_allow_html=True)
        interaction_columns = ['Me gusta', 'Comentarios', 'Veces que se compartió']
        plt.figure(figsize=(3, 1))
        datos[interaction_columns].sum().plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Total de Me gusta, Comentarios y Veces que se compartió')
        plt.xlabel('Tipo de Interacción')
        plt.ylabel('Total')
        fig7, ax7 = plt.gcf(), plt.gca()
        st.pyplot(fig7, ax7, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de barras de tipo de interaccion, muestra la totalidad de interacciones por tipo de interacciones en la totalidad de las publicaciones registradas. En el eje x se muestra diferentes items a considerar siendo los me gusta, comentarios y veces que se compartio, mientras que en el eje y se indica el número de interacciones totales que se tiene en su totalidad.</p>", unsafe_allow_html=True)

def datosFacebook(datos):
    st.markdown("<h2 style='font-size:25px;'><strong>Datos de publicaciones en facebook.</strong></h2>", unsafe_allow_html=True)
    st.write(datos)
    st.markdown("<p style='text-align: justify;'>Esta tabla muestra los datos relacionados con las publicaciones realizadas en la red social de Facebook, los cuales están almacenados en el archivo CSV alojado en Drive.</p>", unsafe_allow_html=True)

def holtWinters(datos):
    #HOLT WINTERS
    # Seleccionar la columna para la predicción
    st.markdown("<h2><strong>HOLT WINTERS</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify;'>El modelo Holt-Winters es una técnica de suavizado exponencial ampliamente utilizada para predecir series temporales con patrones estacionales. Funciona al calcular estimaciones suavizadas de los niveles, tendencias y componentes estacionales presentes en los datos históricos. Este modelo considera tanto la tendencia a corto plazo como la estacionalidad, lo que lo hace especialmente útil para pronosticar variaciones periódicas que puedan surgir en el futuro basandose en el conjunto de datos.</p>", unsafe_allow_html=True)
    serie_temporal = datos['Interacciones']

    # Dividir en conjunto de entrenamiento y conjunto de prueba
    train_size = int(len(serie_temporal) * 0.8)
    train, test = serie_temporal[:train_size], serie_temporal[train_size:]

    # Crear y ajustar el modelo Holt-Winters Exponential Smoothing
    modelo_hw = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)  # Configura los parámetros según sea necesario
    modelo_hw_entrenado = modelo_hw.fit()

    # Obtener los coeficientes del modelo Holt-Winters
    coeficientes = pd.DataFrame(modelo_hw_entrenado.params.items(), columns=['Parámetro', 'Valor'])
    

    # Realizar predicciones en el conjunto de prueba
    predicciones = modelo_hw_entrenado.forecast(len(test))

    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(test, predicciones)

    # Visualizar resultados
    plt.figure(figsize=(3, 1))
    plt.title('Predicción de interacciones utilizando Holt-Winters')
    plt.xlabel('DatosTotales')
    plt.ylabel('Interacciones')
    plt.plot(train.index, train, label='Conjunto de entrenamiento', linewidth=0.5)
    plt.plot(test.index, test, label='Conjunto de prueba', linewidth=0.5)
    plt.plot(test.index, predicciones, label='Predicciones', linewidth=0.5)
    plt.legend()
    st.pyplot(plt, use_container_width=False)
    

    st.write("**Tabla de Coeficientes:**")
    st.write(coeficientes)

def sarima(datos):
    #SARIMA
    st.markdown("<h2><strong>SARIMA</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify;'>El modelo SARIMA (Seasonal Autoregressive Integrated Moving Average) es una técnica utilizada para predecir series temporales. Funciona mediante la identificación de patrones estacionales, tendencias y componentes aleatorios en los datos históricos. A través de la identificación de parámetros como la estacionalidad, SARIMA puede generar pronósticos precisos que tienen en cuenta las tendencias y patrones estacionales presentes en los datos de un conjunto de datos. </p>", unsafe_allow_html=True)
    
    serie_temporal = datos['Interacciones']
    train_size = int(len(serie_temporal) * 0.8)
    train, test = serie_temporal[:train_size], serie_temporal[train_size:]
    modelo_sarima = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 12)) 
    modelo_sarima_entrenado = modelo_sarima.fit()
    coeficientes_html = modelo_sarima_entrenado.summary().tables[1].as_html()
    coeficientes_df = pd.read_html(coeficientes_html, header=0, index_col=0)[0]

    predicciones = modelo_sarima_entrenado.get_forecast(steps=len(test)).predicted_mean
    mse = mean_squared_error(test, predicciones)

    plt.figure(figsize=(3, 1))
    plt.title('Predicción de interacciones utilizando SARIMA')
    plt.xlabel('DatosTotales')
    plt.ylabel('Interacciones totales')
    plt.plot(train.index, train, label='Conjunto de entrenamiento', linewidth=0.5)
    plt.plot(test.index, test, label='Conjunto de prueba', linewidth=0.5)
    plt.plot(test.index, predicciones, label='Predicciones', linewidth=0.5)
    plt.legend()
    st.pyplot(plt, use_container_width=False)
    

    st.write("**Tabla de Coeficientes:**")
    st.write(coeficientes_df)
    st.subheader("Resultados completos de SARIMAX:")
    st.write(modelo_sarima_entrenado.summary())
    st.subheader("Residuos del modelo:")
    residuos = modelo_sarima_entrenado.resid
    st.write(residuos)

def prediccionesFacebook(datos):
    # Dividir la pantalla en dos columnas
    col1, col2 = st.columns(2)

    # Método para Holt-Winters
    with col1:
        sarima(datos)
        

    # Método para SARIMA
    with col2:
        holtWinters(datos)
         
def validacionFacebook(datos):
    # Columnas en Streamlit
    col1, col2 = st.columns(2)

    # VALIDACIÓN CRUZADA SARIMA
    with col1:

        st.markdown("<h2  style='font-size:25px;'><strong>VALIDACIÓN CRUZADA SARIMA</strong></h2>", unsafe_allow_html=True)
        serie_temporal = datos['Interacciones']

        # Definir el número de pliegues para la validación cruzada k-fold
        n_splits = 5

        # Inicializar el objeto TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Inicializar una lista para almacenar los errores cuadráticos medios (MSE) de cada pliegue
        mse_scores = []

        # Definir una paleta de colores para los pliegues
        colors = plt.cm.viridis(np.linspace(0, 1, n_splits))

        for i, (train_index, test_index) in enumerate(tscv.split(serie_temporal), 1):
            # Dividir los datos en conjuntos de entrenamiento y prueba
            train, test = serie_temporal.iloc[train_index], serie_temporal.iloc[test_index]

            # Crear y ajustar el modelo SARIMA
            modelo_sarima = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 12))
            modelo_sarima_entrenado = modelo_sarima.fit()

            # Realizar predicciones en el conjunto de prueba
            predicciones = modelo_sarima_entrenado.get_forecast(steps=len(test)).predicted_mean

            # Calcular el error cuadrático medio (MSE) y agregarlo a la lista de puntuaciones MSE
            mse = mean_squared_error(test, predicciones)
            mse_scores.append(mse)

            st.write(f"Fold {i}: Error Cuadrático Medio (MSE): {mse}")

            # Visualizar resultados
            if i == 1:
                fig, ax = plt.subplots(figsize=(3, 1))  # Definir la figura y los ejes solo en el primer bucle
            ax.plot(test.index, predicciones, label=f'Fold {i} - Predicciones', color=colors[i-1], linestyle='-', linewidth=0.5)
        
        average_mse = np.mean(mse_scores)
        st.write("Error cuadrático medio promedio:", average_mse)

        # Mostrar la gráfica
        plt.plot(serie_temporal.index, serie_temporal, label='Serie Temporal', color='gray', linestyle='-', linewidth=0.5)
        ax.set_title('Predicciones de Interacciones utilizando SARIMA con Validación Cruzada')
        ax.set_xlabel('Total de publicaciones')
        ax.set_ylabel('Interacciones totales')
        ax.legend()
        st.pyplot(fig, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>Este es un texto de ejemplo para explicar los resultados de la validación cruzada SARIMA.</p>", unsafe_allow_html=True)

    # VALIDACIÓN CRUZADA HOLT WINTERS
    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>VALIDACIÓN CRUZADA HOLT WINTERS</strong></h2>", unsafe_allow_html=True)
        # Seleccionar la columna para la predicción
        serie_temporal = datos['Interacciones']

        # Definir el número de pliegues para la validación cruzada k-fold
        n_splits = 5

        # Inicializar el objeto TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Inicializar una lista para almacenar los errores cuadráticos medios (MSE) de cada pliegue
        mse_scores = []

        # Inicializar la figura para combinar todas las gráficas de los pliegues
        plt.figure(figsize=(3, 1))

        # Inicializar listas para almacenar predicciones y errores de cada fold
        all_predictions = []
        all_errors = []

        # Iterar sobre los pliegues generados por TimeSeriesSplit
        for fold_index, (train_index, test_index) in enumerate(tscv.split(serie_temporal)):
            # Dividir los datos en conjuntos de entrenamiento y prueba
            train, test = serie_temporal.iloc[train_index], serie_temporal.iloc[test_index]

            # Crear y ajustar el modelo Holt-Winters Exponential Smoothing
            modelo_hw = ExponentialSmoothing(train, seasonal='add', seasonal_periods=6)
            modelo_hw_entrenado = modelo_hw.fit()

            # Realizar predicciones en el conjunto de prueba
            predicciones = modelo_hw_entrenado.forecast(len(test))

            # Calcular el error cuadrático medio (MSE) y agregarlo a la lista de puntuaciones MSE
            mse = mean_squared_error(test, predicciones)
            mse_scores.append(mse)

            # Almacenar predicciones y errores
            all_predictions.append(predicciones)
            all_errors.append(mse)

            # Agregar la gráfica del fold actual a la figura combinada
            plt.plot(test.index, predicciones, label=f'Fold {fold_index + 1}', linewidth=0.5)

        # Mostrar predicciones y errores para cada fold
        for fold_index, (predicciones, error) in enumerate(zip(all_predictions, all_errors), start=1):
            st.write(f"Fold {fold_index}: Error Cuadrático Medio (MSE): {error}")

        # Calcular el error cuadrático medio promedio (MSE) de todos los pliegues
        average_mse = np.mean(mse_scores)
        st.write("Error cuadrático medio promedio:", average_mse)

        # Visualizar conjunto de entrenamiento y prueba
        plt.plot(serie_temporal.index, serie_temporal, label='Serie Temporal', color='gray', linestyle='-', linewidth=0.5)
        plt.xlabel('Total de publicaciones')
        plt.ylabel('Interacciones totales')
        plt.title('Predicción de interacciones utilizando Holt-Winters con Validación Cruzada')
        plt.legend()
        fig2, ax2 = plt.gcf(), plt.gca() 
        st.pyplot(fig2, ax2, use_container_width=False) 

        st.markdown("<p style='text-align: justify;'>Este es un texto de ejemplo para explicar los resultados de la validación cruzada Holt-Winters.</p>", unsafe_allow_html=True)


#FUNCIONES INSTRAGRAM
def analisis_instagram(datosInstagram):
    datos_numericos = datosInstagram.select_dtypes(include=['float64', 'int64'])

    # Dividir la pantalla en dos columnas
    col1, col2 = st.columns(2)

    # GRAFICO 1: Correlación
    with col1:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE CORRELACIÓN</strong></h2>", unsafe_allow_html=True)
        correlation_matrix = datos_numericos.corr()
        fig1, ax1 = plt.subplots(figsize=(2, 2))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax1, square=True)
        ax1.set_title('Matriz de Correlación')
        plt.gcf().patch.set_linewidth(2)
        plt.gcf().patch.set_edgecolor('black')
        st.pyplot(fig1, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de correlación relaciona las variables del conjunto de datos. Es una tabla que muestra la correlación entre cada par de variables. Estos coeficientes van de -1 a 1, indicando la fuerza y dirección de la relación entre las variables. Un valor de 1 implica una correlación positiva perfecta, -1 una correlación negativa perfecta y 0 ausencia de relación.</p>", unsafe_allow_html=True)

    # GRAFICO 2: Barras de tipo de interacción
    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE BARRAS TIPO DE INTERACCIÓN</strong></h2>", unsafe_allow_html=True)
        interaction_columns = ['Me gusta', 'Comentarios', 'Veces que se compartió']
        fig2, ax2 = plt.subplots(figsize=(3, 1))
        datosInstagram[interaction_columns].sum().plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Total de Me gusta, Comentarios y Veces que se compartió')
        plt.xlabel('Tipo de Interacción')
        plt.ylabel('Total')
        st.pyplot(fig2, ax2, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de barras de interacciones, muestra la totalidad de interacciones por tipo de publicación. En el eje x se muestra el tipo de publicación, mientras que en el eje y se indica el número de interacciones totales que se tiene por publicacion.</p>", unsafe_allow_html=True)

    # GRAFICO 3: Caja de Me gusta por mes
    with col1:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE CAJA DE ME GUSTA POR MES</strong></h2>", unsafe_allow_html=True)
        datosInstagram['Hora de publicación'] = pd.to_datetime(datosInstagram['Hora de publicación'])
        datosInstagram['Año'] = datosInstagram['Hora de publicación'].dt.year
        datosInstagram['Mes'] = datosInstagram['Hora de publicación'].dt.month
        datosInstagram['Nombre_Mes'] = datosInstagram['Mes'].apply(lambda x: calendar.month_name[x])
        datosInstagram['Mes_Y_Año'] = datosInstagram['Nombre_Mes'] + ' ' + datosInstagram['Año'].astype(str)
        fig3, ax3 = plt.subplots(figsize=(3, 1))
        sns.boxplot(x='Mes_Y_Año', y='Me gusta', data=datosInstagram, palette='viridis')
        plt.title('Diagrama de Caja: Me gusta vs Mes')
        plt.xlabel('Fecha')
        plt.ylabel('Me Gusta')
        plt.xticks(rotation=45)
        st.pyplot(fig3, ax3, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de caja de me gusta presenta información sobre el número de interacciones según el mes de publicación. En el eje x se divide en meses, mientras que en el eje y se indica el número de interacciones.</p>", unsafe_allow_html=True)

    # GRAFICO 4: Caja de interacciones
    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE CAJA DE INTERACCIONES</strong></h2>", unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(3, 1))
        sns.boxplot(x='Tipo de publicación', y='Me gusta', data=datosInstagram, palette='viridis')
        plt.title('Interacciones por Tipo de Publicación')
        plt.xlabel('Tipo de Publicación')
        plt.ylabel('Interacciones')
        st.pyplot(fig4, ax4, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de caja de interaciones presenta información sobre el número de interacciones según el tipo de publicación. En el eje x se muestra el tipo de publicacion devidido por me gustas, real de instagram y secuencias de instagram, mientras que en el eje y se indica el número de interacciones que se reflejan en el conjunto de datos.</p>", unsafe_allow_html=True)

    # GRAFICO 5: Barras para alcance mensual
    with col1:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE BARRAS PARA ALCANCE MENSUAL</strong></h2>", unsafe_allow_html=True)
        datosInstagram['Hora de publicación'] = pd.to_datetime(datosInstagram['Hora de publicación'])
        datosInstagram['Año'] = datosInstagram['Hora de publicación'].dt.year
        datosInstagram['Mes'] = datosInstagram['Hora de publicación'].dt.month
        datosInstagram['Nombre_Mes'] = datosInstagram['Mes'].apply(lambda x: calendar.month_name[x])
        datosInstagram['Mes_Y_Año'] = datosInstagram['Nombre_Mes'] + ' ' + datosInstagram['Año'].astype(str)
        fig5, ax5 = plt.subplots(figsize=(3, 1))
        sns.barplot(x='Mes_Y_Año', y='Alcance', data=datosInstagram, palette='viridis', ci=None)
        plt.title('Diagrama de barras de Alcance por meses')
        plt.xlabel('Fecha')
        plt.ylabel('Alcance')
        plt.xticks(rotation=45)
        st.pyplot(fig5, ax5, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de barras de alcance mensual presenta información sobre el alcance total que se obtuvo a travez del tiempo. En el eje x se muestra los meses del conjunto de datos, mientras que en el eje y se presenta el alcance total que se obtuvo.</p>", unsafe_allow_html=True)

def datosInsta(datosInstagram):
    st.markdown("<h2 style='font-size:25px;'><strong>Datos de publicaciones en instagram.</strong></h2>", unsafe_allow_html=True)
    st.write(datosInstagram)
    st.markdown("<p style='text-align: justify;'>Esta tabla muestra los datos relacionados con las publicaciones realizadas en la red social de Instagram, los cuales están almacenados en el archivo CSV alojado en Drive.</p>", unsafe_allow_html=True)
    
def holtWintersInstagram(datosInstagram):
    #HOLT WINTERS
    # Seleccionar la columna para la predicción
    st.markdown("<h2><strong>HOLT WINTERS</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify;'>El modelo Holt-Winters es una técnica de suavizado exponencial ampliamente utilizada para predecir series temporales con patrones estacionales. Funciona al calcular estimaciones suavizadas de los niveles, tendencias y componentes estacionales presentes en los datos históricos. Este modelo considera tanto la tendencia a corto plazo como la estacionalidad, lo que lo hace especialmente útil para pronosticar variaciones periódicas que puedan surgir en el futuro basandose en el conjunto de datos.</p>", unsafe_allow_html=True)
    
    serie_temporal = datosInstagram['Impresiones']

    # Dividir en conjunto de entrenamiento y conjunto de prueba
    train_size = int(len(serie_temporal) * 0.8)
    train, test = serie_temporal[:train_size], serie_temporal[train_size:]

    # Crear y ajustar el modelo Holt-Winters Exponential Smoothing
    modelo_hw = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)  # Configura los parámetros según sea necesario
    modelo_hw_entrenado = modelo_hw.fit()

    # Obtener los coeficientes del modelo Holt-Winters
    coeficientes = pd.DataFrame(modelo_hw_entrenado.params.items(), columns=['Parámetro', 'Valor'])
    

    # Realizar predicciones en el conjunto de prueba
    predicciones = modelo_hw_entrenado.forecast(len(test))

    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(test, predicciones)

    # Visualizar resultados
    plt.figure(figsize=(3, 1))
    plt.title('Predicción de las impresiones utilizando Holt-Winters')
    plt.xlabel('Datos Totales')
    plt.ylabel('Impresiones Totales')
    plt.plot(train.index, train, label='Conjunto de entrenamiento', linewidth=0.5)
    plt.plot(test.index, test, label='Conjunto de prueba', linewidth=0.5)
    plt.plot(test.index, predicciones, label='Predicciones', linewidth=0.5)
    plt.legend()
    st.pyplot(plt, use_container_width=False)
    

    st.write("**Tabla de Coeficientes:**")
    st.write(coeficientes)

def sarimaInstagram(datosInstagram):
    st.markdown("<h2><strong>SARIMA</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify;'>El modelo SARIMA (Seasonal Autoregressive Integrated Moving Average) es una técnica utilizada para predecir series temporales. Funciona mediante la identificación de patrones estacionales, tendencias y componentes aleatorios en los datos históricos. A través de la identificación de parámetros como la estacionalidad, SARIMA puede generar pronósticos precisos que tienen en cuenta las tendencias y patrones estacionales presentes en los datos de un conjunto de datos.</p>", unsafe_allow_html=True)
    
    serie_temporal = datosInstagram['Impresiones']
    train_size = int(len(serie_temporal) * 0.8)
    train, test = serie_temporal[:train_size], serie_temporal[train_size:]
    modelo_sarima = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 12)) 
    modelo_sarima_entrenado = modelo_sarima.fit()
    coeficientes_html = modelo_sarima_entrenado.summary().tables[1].as_html()
    coeficientes_df = pd.read_html(coeficientes_html, header=0, index_col=0)[0]
    
    predicciones = modelo_sarima_entrenado.get_forecast(steps=len(test)).predicted_mean
    mse = mean_squared_error(test, predicciones)

    plt.figure(figsize=(3, 1))
    plt.title('Predicción de las impresiones utilizando SARIMA')
    plt.xlabel('Datos Totales')
    plt.ylabel('Impresiones totales')
    plt.plot(train.index, train, label='Conjunto de entrenamiento', linewidth=0.5)
    plt.plot(test.index, test, label='Conjunto de prueba', linewidth=0.5)
    plt.plot(test.index, predicciones, label='Predicciones', linewidth=0.5)
    plt.legend()
    st.pyplot(plt, use_container_width=False)
    

    st.write("**Tabla de Coeficientes:**")
    st.write(coeficientes_df)
    st.subheader("Resultados completos de SARIMAX:")
    st.write(modelo_sarima_entrenado.summary())
    st.subheader("Residuos del modelo:")
    residuos = modelo_sarima_entrenado.resid
    st.write(residuos)

def prediccionesInstagram(datosInstagram):
    # Dividir la pantalla en dos columnas
    col1, col2 = st.columns(2)

    # Método para Holt-Winters
    with col1:
        sarimaInstagram(datosInstagram)
        
    # Método para SARIMA
    with col2:
        holtWintersInstagram(datosInstagram) 

def validacionInstagram(datosInstagram):
    # Columnas en Streamlit
    col1, col2 = st.columns(2)

    # VALIDACIÓN CRUZADA SARIMA
    with col1:
        st.markdown("<h2 style='font-size:25px;'><strong>VALIDACIÓN CRUZADA SARIMA</strong></h2>", unsafe_allow_html=True)
        serie_temporal = datosInstagram['Alcance']

        # Definir el número de pliegues para la validación cruzada k-fold
        n_splits = 5

        # Inicializar el objeto TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Inicializar una lista para almacenar los errores cuadráticos medios (MSE) de cada pliegue
        mse_scores = []

        # Inicializar figuras y ejes para la visualización
        fig, ax = plt.subplots(figsize=(3, 1))

        # Definir una paleta de colores para los pliegues
        colors = plt.cm.viridis(np.linspace(0, 1, n_splits))

        for i, (train_index, test_index) in enumerate(tscv.split(serie_temporal), 1):
            # Dividir los datos en conjuntos de entrenamiento y prueba
            train, test = serie_temporal.iloc[train_index], serie_temporal.iloc[test_index]

            # Crear y ajustar el modelo SARIMA
            modelo_sarima = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 12))  # Ajusta el orden según sea necesario
            modelo_sarima_entrenado = modelo_sarima.fit()

            # Realizar predicciones en el conjunto de prueba
            predicciones = modelo_sarima_entrenado.get_forecast(steps=len(test)).predicted_mean

            # Calcular el error cuadrático medio (MSE) y agregarlo a la lista de puntuaciones MSE
            mse = mean_squared_error(test, predicciones)
            mse_scores.append(mse)

            st.write(f"Fold {i}: Error Cuadrático Medio (MSE): {mse}")

            # Visualizar resultados
            ax.plot(test.index, predicciones, label=f'Fold {i} - Predicciones', color=colors[i-1], linestyle='-', linewidth=0.5)
        average_mse = np.mean(mse_scores)
        st.write("Error cuadrático medio promedio:", average_mse)

        # Mostrar la gráfica
        ax.plot(serie_temporal.index, serie_temporal, label='Serie Temporal', color='gray', linestyle='-', linewidth=0.5)
        ax.set_title('Predicciones de Alcance utilizando SARIMA con Validación Cruzada')
        ax.set_xlabel('Total de publicaciones')
        ax.set_ylabel('Alcance total')
        ax.legend()
        st.pyplot(fig, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>Este es un texto de ejemplo para explicar los resultados de la validación cruzada SARIMA.</p>", unsafe_allow_html=True)

    # VALIDACIÓN CRUZADA HOLT WINTERS
    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>VALIDACIÓN CRUZADA HOLT WINTERS</strong></h2>", unsafe_allow_html=True)
        # Seleccionar la columna para la predicción
        serie_temporal = datosInstagram['Alcance']

        # Definir el número de pliegues para la validación cruzada k-fold
        n_splits = 5

        # Inicializar el objeto TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Inicializar una lista para almacenar los errores cuadráticos medios (MSE) de cada pliegue
        mse_scores = []

        # Inicializar la figura para combinar todas las gráficas de los pliegues
        plt.figure(figsize=(3, 1))

        # Inicializar listas para almacenar predicciones y errores de cada fold
        all_predictions = []
        all_errors = []

        # Iterar sobre los pliegues generados por TimeSeriesSplit
        for fold_index, (train_index, test_index) in enumerate(tscv.split(serie_temporal)):
            # Dividir los datos en conjuntos de entrenamiento y prueba
            train, test = serie_temporal.iloc[train_index], serie_temporal.iloc[test_index]

            # Crear y ajustar el modelo Holt-Winters Exponential Smoothing
            modelo_hw = ExponentialSmoothing(train, seasonal='add', seasonal_periods=3)  # Configura los parámetros según sea necesario
            modelo_hw_entrenado = modelo_hw.fit()

            # Realizar predicciones en el conjunto de prueba
            predicciones = modelo_hw_entrenado.forecast(len(test))

            # Calcular el error cuadrático medio (MSE) y agregarlo a la lista de puntuaciones MSE
            mse = mean_squared_error(test, predicciones)
            mse_scores.append(mse)

            # Almacenar predicciones y errores
            all_predictions.append(predicciones)
            all_errors.append(mse)

            # Agregar la gráfica del fold actual a la figura combinada
            plt.plot(test.index, predicciones, label=f'Fold {fold_index + 1}', linewidth=0.5)

        # Mostrar predicciones y errores para cada fold
        for fold_index, (predicciones, error) in enumerate(zip(all_predictions, all_errors), start=1):
            st.write(f"Fold {fold_index}: Error Cuadrático Medio (MSE): {error}")

        # Calcular el error cuadrático medio promedio (MSE) de todos los pliegues
        average_mse = np.mean(mse_scores)
        st.write("Error cuadrático medio promedio:", average_mse)

        # Visualizar conjunto de entrenamiento y prueba
        plt.plot(serie_temporal.index, serie_temporal, label='Serie Temporal', color='gray', linestyle='-', linewidth=0.5)
        plt.xlabel('Total de publicaciones')
        plt.ylabel('Alcance total')
        plt.title('Predicción de Alcance utilizando Holt-Winters con Validación Cruzada')
        plt.legend()
        fig2, ax2 = plt.gcf(), plt.gca() 
        st.pyplot(fig2, ax2, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>Este es un texto de ejemplo para explicar los resultados de la validación cruzada Holt-Winters.</p>", unsafe_allow_html=True)

#FUNCIONES METRICAS
def analisis_metricas(datosMetricas):
    # Columnas en Streamlit
    col1, col2 = st.columns(2)

    # GRAFICO 1: Matriz de Correlación
    with col1:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO DE CORRELACIÓN</strong></h2>", unsafe_allow_html=True)
        correlation_matrix = datosMetricas.select_dtypes(include=['float64', 'int64']).corr()
        fig1, ax1 = plt.subplots(figsize=(2, 2))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax1, square=True)
        ax1.set_title('Matriz de Correlación')
        ax1.set_xlabel('Fecha (Mes Año)', fontsize=4, fontname='Arial')  # Cambiar el tipo de letra a Arial y ajustar el tamaño de la fuente en el eje x
        st.pyplot(fig1, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El gráfico de correlación relaciona las variables del conjunto de datos. Es una tabla que muestra la correlación entre cada par de variables. Estos coeficientes van de -1 a 1, indicando la fuerza y dirección de la relación entre las variables. Un valor de 1 implica una correlación positiva perfecta, -1 una correlación negativa perfecta y 0 ausencia de relación.</p>", unsafe_allow_html=True)

    # GRAFICO 2: Comparativa de Alcance entre Facebook e Instagram
    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO COMPARATIVO DE ALCANCE ENTRE FACEBOOK E INSTAGRAM</strong></h2>", unsafe_allow_html=True)
        datosMetricas['Fecha'] = pd.to_datetime(datosMetricas['Fecha'])
        primeros_dias_mes = datosMetricas['Fecha'].dt.to_period('M').dt.start_time
        fig2, ax2 = plt.subplots(figsize=(3, 1))
        ax2.plot(datosMetricas['Fecha'], datosMetricas['Alcance_Facebook'], label='Alcance en Facebook', linestyle='-', linewidth=0.5)
        ax2.plot(datosMetricas['Fecha'], datosMetricas['Alcance_Instagram'], label='Alcance en Instagram', linestyle='-', linewidth=0.5)
        ax2.set_title('Comparativa de Tendencia del Alcance en Facebook e Instagram')
        ax2.set_xlabel('Fecha (Mes Año)', fontsize=4, fontname='Arial')  # Cambiar el tipo de letra a Arial y ajustar el tamaño de la fuente en el eje x
        ax2.set_ylabel('Alcance')
        ax2.legend()
        ax2.set_xticks(primeros_dias_mes)
        ax2.set_xticklabels([date.strftime('%B %Y') for date in primeros_dias_mes], rotation=45, fontsize=4, fontname='Arial', fontweight='light')  # Establecer el formato del eje x, el tamaño de la fuente y el tipo de letra
        ax2.tick_params(axis='y', which='both', width=0.3)  # Grosor de las líneas en el eje y
        st.pyplot(fig2, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El grafico comparativo de tendecia entre facebook vs intagram compara la tendecia del alcance de las dos redes sociales. En el eje x se presenta las fechas divididas mensualmente, mientras que en el eje y se presenta el alcance total que se obtuvo.</p>", unsafe_allow_html=True)    
    
    # GRAFICO 3: Comparativa de Vistas entre Facebook e Instagram
    with col1:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO COMPARATIVO DE VISTAS ENTRE FACEBOOK E INSTAGRAM</strong></h2>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(3, 1))
        ax3.plot(datosMetricas['Fecha'], datosMetricas['Vistas_Facebook'], label='Visitas en Facebook', linestyle='-', linewidth=0.5)
        ax3.plot(datosMetricas['Fecha'], datosMetricas['Vistas_Instagram'], label='Visitas en Instagram', linestyle='-', linewidth=0.5)
        ax3.set_title('Comparativa de Tendencia de Visitas en Facebook e Instagram')
        ax3.set_xlabel('Fecha (Mes Año)', fontsize=4, fontname='Arial')  # Cambiar el tipo de letra a Arial y ajustar el tamaño de la fuente en el eje x
        ax3.set_ylabel('Visitas')
        ax3.legend()
        ax3.set_xticks(primeros_dias_mes)
        ax3.set_xticklabels([date.strftime('%B %Y') for date in primeros_dias_mes], rotation=45, fontsize=4, fontname='Arial', fontweight='light')  # Establecer el formato del eje x, el tamaño de la fuente y el tipo de letra
        ax3.tick_params(axis='y', which='both', width=0.3)
        st.pyplot(fig3, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El grafico comparativo de vistas entre facebook vs intagram compara las vistas que genero cada publicacion de las dos redes sociales. En el eje x se presenta las fechas divididas mensualmente, mientras que en el eje y se presenta las vistas total que se obtuvieron por publicación.</p>", unsafe_allow_html=True)    

    # GRAFICO 4: Comparativa de Seguidores entre Facebook e Instagram
    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>GRÁFICO COMPARATIVO DE SEGUIDORES ENTRE FACEBOOK E INSTAGRAM</strong></h2>", unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(3, 1))
        ax4.plot(datosMetricas['Fecha'], datosMetricas['Seguidores_Facebook'], label='Seguidores en Facebook', linestyle='-', linewidth=0.5)
        ax4.plot(datosMetricas['Fecha'], datosMetricas['Seguidores_Instagram'], label='Seguidores en Instagram', linestyle='-', linewidth=0.5)
        ax4.set_title('Comparativa de Tendencia de Seguidores en Facebook e Instagram')
        ax4.set_xlabel('Fecha (Mes Año)', fontsize=4, fontname='Arial')  # Cambiar el tipo de letra a Arial y ajustar el tamaño de la fuente en el eje x
        ax4.set_ylabel('Seguidores')
        ax4.legend()
        ax4.set_xticks(primeros_dias_mes)
        ax4.set_xticklabels([date.strftime('%B %Y') for date in primeros_dias_mes], rotation=45, fontsize=4, fontname='Arial', fontweight='light')  # Establecer el formato del eje x, el tamaño de la fuente y el tipo de letra
        ax4.tick_params(axis='y', which='both', width=0.3)
        st.pyplot(fig4, use_container_width=False)
        st.markdown("<p style='text-align: justify;'>El grafico comparativo de seguidores entre facebook vs intagram compara la cantidad de seguidores alcanzadas de las dos redes sociales. En el eje x se presenta las fechas divididas mensualmente, mientras que en el eje y se presenta el total de seguidores obtenido.</p>", unsafe_allow_html=True)    

def dataMetricas(datosMetricas):
    st.markdown("<h2 style='font-size:25px;'><strong>Datos de las metricas de facebook e instagram.</strong></h2>", unsafe_allow_html=True)
    st.write(datosMetricas)
    st.markdown("<p style='text-align: justify;'>Esta tabla muestra los datos relacionados con las metricas realizadas en las redes sociales de facebook e instagram, los cuales están almacenados en el archivo CSV alojado en Drive.</p>", unsafe_allow_html=True)

def holtWintersMetricas(datosMetricas):
    #HOLT WINTERS
    st.markdown("<h2><strong>HOLT WINTERS</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify;'>El modelo Holt-Winters es una técnica de suavizado exponencial ampliamente utilizada para predecir series temporales con patrones estacionales. Funciona al calcular estimaciones suavizadas de los niveles, tendencias y componentes estacionales presentes en los datos históricos. Este modelo considera tanto la tendencia a corto plazo como la estacionalidad, lo que lo hace especialmente útil para pronosticar variaciones periódicas que puedan surgir en el futuro basandose en el conjunto de datos.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3><strong>HOLT WINTERS PREDICCIÓN INSTAGRAM</strong></h3>", unsafe_allow_html=True)
    serie_temporal = datosMetricas['Alcance_Instagram']

    # Dividir en conjunto de entrenamiento y conjunto de prueba
    train_size = int(len(serie_temporal) * 0.8)
    train, test = serie_temporal[:train_size], serie_temporal[train_size:]

    # Crear y ajustar el modelo Holt-Winters Exponential Smoothing
    modelo_hw = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)  # Configura los parámetros según sea necesario
    modelo_hw_entrenado = modelo_hw.fit()

    # Obtener los coeficientes del modelo Holt-Winters
    coeficientes = pd.DataFrame(modelo_hw_entrenado.params.items(), columns=['Parámetro', 'Valor'])
    

    # Realizar predicciones en el conjunto de prueba
    predicciones = modelo_hw_entrenado.forecast(len(test))

    # Calcular el error cuadrático medio (MSE)
    mse1 = mean_squared_error(test, predicciones)

    # Visualizar resultados
    plt.figure(figsize=(3, 1))
    plt.title('Predicción de Alcance utilizando Holt-Winters')
    plt.xlabel('Datos Totales')
    plt.ylabel('Alcance Total')
    plt.plot(train.index, train, label='Conjunto de entrenamiento', linewidth=0.5)
    plt.plot(test.index, test, label='Conjunto de prueba', linewidth=0.5)
    plt.plot(test.index, predicciones, label='Predicciones', linewidth=0.5)
    plt.legend()
    fig1, ax1 = plt.gcf(), plt.gca()
    st.pyplot(fig1, ax1, use_container_width=False)
    

    st.write("**Tabla de Coeficientes:**")
    st.write(coeficientes)



    # Seleccionar la columna para la predicción
    st.markdown("<h2><strong>HOLT WINTERS</strong></h2>", unsafe_allow_html=True)
    st.markdown("<h3><strong>HOLT WINTERS PREDICCIÓN FACEBOOK</strong></h3>", unsafe_allow_html=True)
    serie_temporal = datosMetricas['Alcance_Facebook']

    # Dividir en conjunto de entrenamiento y conjunto de prueba
    train_size = int(len(serie_temporal) * 0.8)
    train, test = serie_temporal[:train_size], serie_temporal[train_size:]

    # Crear y ajustar el modelo Holt-Winters Exponential Smoothing
    modelo_hw = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)  # Configura los parámetros según sea necesario
    modelo_hw_entrenado = modelo_hw.fit()

    # Obtener los coeficientes del modelo Holt-Winters
    coeficientes = pd.DataFrame(modelo_hw_entrenado.params.items(), columns=['Parámetro', 'Valor'])
    

    # Realizar predicciones en el conjunto de prueba
    predicciones = modelo_hw_entrenado.forecast(len(test))

    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(test, predicciones)

    # Visualizar resultados
    plt.figure(figsize=(3, 1))
    plt.title('Predicción de Alcance utilizando Holt-Winters')
    plt.xlabel('Datos Totales')
    plt.ylabel('Alcance Total')
    plt.plot(train.index, train, label='Conjunto de entrenamiento', linewidth=0.5)
    plt.plot(test.index, test, label='Conjunto de prueba', linewidth=0.5)
    plt.plot(test.index, predicciones, label='Predicciones', linewidth=0.5)
    plt.legend()
    st.pyplot(plt, use_container_width=False)
    

    st.write("**Tabla de Coeficientes:**")
    st.write(coeficientes)

def sarimaMetricas(datosMetricas):
    st.markdown("<h2><strong>SARIMA</strong></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify;'>El modelo SARIMA (Seasonal Autoregressive Integrated Moving Average) es una técnica utilizada para predecir series temporales. Funciona mediante la identificación de patrones estacionales, tendencias y componentes aleatorios en los datos históricos. A través de la identificación de parámetros como la estacionalidad, SARIMA puede generar pronósticos precisos que tienen en cuenta las tendencias y patrones estacionales presentes en los datos de un conjunto de datos.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3><strong>SARIMA PREDICCIÓN INSTAGRAM</strong></h3>", unsafe_allow_html=True)
    serie_temporal = datosMetricas['Alcance_Instagram']
    train_size = int(len(serie_temporal) * 0.8)
    train, test = serie_temporal[:train_size], serie_temporal[train_size:]
    modelo_sarima = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 12)) 
    modelo_sarima_entrenado = modelo_sarima.fit()
    coeficientes_html = modelo_sarima_entrenado.summary().tables[1].as_html()
    coeficientes_df = pd.read_html(coeficientes_html, header=0, index_col=0)[0]
    
    predicciones = modelo_sarima_entrenado.get_forecast(steps=len(test)).predicted_mean
    mse = mean_squared_error(test, predicciones)

    plt.figure(figsize=(3, 1))
    plt.title('Predicción de Alcance utilizando SARIMA')
    plt.xlabel('Datos Totales')
    plt.ylabel('Alcance Total')
    plt.plot(train.index, train, label='Conjunto de entrenamiento', linewidth=0.5)
    plt.plot(test.index, test, label='Conjunto de prueba', linewidth=0.5)
    plt.plot(test.index, predicciones, label='Predicciones', linewidth=0.5)
    plt.legend()
    st.pyplot(plt, use_container_width=False)
    

    st.write("**Tabla de Coeficientes:**")
    st.write(coeficientes_df)
    st.subheader("Resultados completos de SARIMAX:")
    st.write(modelo_sarima_entrenado.summary())
    st.subheader("Residuos del modelo:")
    residuos = modelo_sarima_entrenado.resid
    st.write(residuos)


    st.markdown("<h3><strong>SARIMA PREDICCIÓN FACEBOOK</strong></h3>", unsafe_allow_html=True)
    serie_temporal = datosMetricas['Alcance_Facebook']
    train_size = int(len(serie_temporal) * 0.8)
    train, test = serie_temporal[:train_size], serie_temporal[train_size:]
    modelo_sarima = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 12)) 
    modelo_sarima_entrenado = modelo_sarima.fit()
    coeficientes_html = modelo_sarima_entrenado.summary().tables[1].as_html()
    coeficientes_df = pd.read_html(coeficientes_html, header=0, index_col=0)[0]
    
    predicciones = modelo_sarima_entrenado.get_forecast(steps=len(test)).predicted_mean
    mse1 = mean_squared_error(test, predicciones)

    plt.figure(figsize=(3, 1))
    plt.title('Predicción de Alcance utilizando SARIMA')
    plt.xlabel('Datos Totales')
    plt.ylabel('Alcance Total')
    plt.plot(train.index, train, label='Conjunto de entrenamiento', linewidth=0.5)
    plt.plot(test.index, test, label='Conjunto de prueba', linewidth=0.5)
    plt.plot(test.index, predicciones, label='Predicciones', linewidth=0.5)
    plt.legend()
    fig1, ax1 = plt.gcf(), plt.gca()
    st.pyplot(fig1, ax1, use_container_width=False)
    

    st.write("**Tabla de Coeficientes:**")
    st.write(coeficientes_df)
    st.subheader("Resultados completos de SARIMAX:")
    st.write(modelo_sarima_entrenado.summary())
    st.subheader("Residuos del modelo:")
    residuos = modelo_sarima_entrenado.resid
    st.write(residuos)

def prediccionesMetricas(datosMetricas):
    # Dividir la pantalla en dos columnas
    col1, col2 = st.columns(2)

    # Método para Holt-Winters
    with col1:
        sarimaMetricas(datosMetricas)
        
    # Método para SARIMA
    with col2:
        holtWintersMetricas(datosMetricas) 

def validacionMetricas(datosMetricas):
    # Columnas en Streamlit
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h2  style='font-size:25px;'><strong>VALIDACIÓN CRUZADA SARIMA INSTAGRAM</strong></h2>", unsafe_allow_html=True)
        serie_temporal = datosMetricas['Alcance_Instagram']

        # Definir el número de pliegues para la validación cruzada k-fold
        n_splits = 5

        # Inicializar el objeto TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Inicializar una lista para almacenar los errores cuadráticos medios (MSE) de cada pliegue
        mse_scores = []

        # Definir una paleta de colores para los pliegues
        colors = plt.cm.viridis(np.linspace(0, 1, n_splits))

        for i, (train_index, test_index) in enumerate(tscv.split(serie_temporal), 1):
            # Dividir los datos en conjuntos de entrenamiento y prueba
            train, test = serie_temporal.iloc[train_index], serie_temporal.iloc[test_index]

            # Crear y ajustar el modelo SARIMA
            modelo_sarima = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 12))
            modelo_sarima_entrenado = modelo_sarima.fit()

            # Realizar predicciones en el conjunto de prueba
            predicciones = modelo_sarima_entrenado.get_forecast(steps=len(test)).predicted_mean

            # Calcular el error cuadrático medio (MSE) y agregarlo a la lista de puntuaciones MSE
            mse = mean_squared_error(test, predicciones)
            mse_scores.append(mse)

            st.write(f"Fold {i}: Error Cuadrático Medio (MSE): {mse}")

            # Visualizar resultados
            if i == 1:
                fig, ax = plt.subplots(figsize=(3, 1))  # Definir la figura y los ejes solo en el primer bucle
            ax.plot(test.index, predicciones, label=f'Fold {i} - Predicciones', color=colors[i-1], linestyle='-', linewidth=0.5)
        
        average_mse = np.mean(mse_scores)
        st.write("Error cuadrático medio promedio:", average_mse)

        # Mostrar la gráfica
        plt.plot(serie_temporal.index, serie_temporal, label='Serie Temporal', color='gray', linestyle='-', linewidth=0.5)
        ax.set_title('Predicciones del alcance utilizando SARIMA con Validación Cruzada')
        ax.set_xlabel('Total de publicaciones')
        ax.set_ylabel('Alcance total')
        ax.legend()
        st.pyplot(fig, use_container_width=False)
        

    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>VALIDACIÓN CRUZADA HOLT WINTERS INSTAGRAM</strong></h2>", unsafe_allow_html=True)
        # Seleccionar la columna para la predicción
        serie_temporal = datosMetricas['Alcance_Instagram']

        # Definir el número de pliegues para la validación cruzada k-fold
        n_splits = 5

        # Inicializar el objeto TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Inicializar una lista para almacenar los errores cuadráticos medios (MSE) de cada pliegue
        mse_scores = []

        # Inicializar la figura para combinar todas las gráficas de los pliegues
        plt.figure(figsize=(3, 1))

        # Inicializar listas para almacenar predicciones y errores de cada fold
        all_predictions = []
        all_errors = []

        # Iterar sobre los pliegues generados por TimeSeriesSplit
        for fold_index, (train_index, test_index) in enumerate(tscv.split(serie_temporal)):
            # Dividir los datos en conjuntos de entrenamiento y prueba
            train, test = serie_temporal.iloc[train_index], serie_temporal.iloc[test_index]

            # Crear y ajustar el modelo Holt-Winters Exponential Smoothing
            modelo_hw = ExponentialSmoothing(train, seasonal='add', seasonal_periods=6)
            modelo_hw_entrenado = modelo_hw.fit()

            # Realizar predicciones en el conjunto de prueba
            predicciones = modelo_hw_entrenado.forecast(len(test))

            # Calcular el error cuadrático medio (MSE) y agregarlo a la lista de puntuaciones MSE
            mse = mean_squared_error(test, predicciones)
            mse_scores.append(mse)

            # Almacenar predicciones y errores
            all_predictions.append(predicciones)
            all_errors.append(mse)

            # Agregar la gráfica del fold actual a la figura combinada
            plt.plot(test.index, predicciones, label=f'Fold {fold_index + 1}', linewidth=0.5)

        # Mostrar predicciones y errores para cada fold
        for fold_index, (predicciones, error) in enumerate(zip(all_predictions, all_errors), start=1):
            st.write(f"Fold {fold_index}: Error Cuadrático Medio (MSE): {error}")

        # Calcular el error cuadrático medio promedio (MSE) de todos los pliegues
        average_mse = np.mean(mse_scores)
        st.write("Error cuadrático medio promedio:", average_mse)

        # Visualizar conjunto de entrenamiento y prueba
        plt.plot(serie_temporal.index, serie_temporal, label='Serie Temporal', color='gray', linestyle='-', linewidth=0.5)
        plt.xlabel('Total de publicaciones')
        plt.ylabel('Alcance total')
        plt.title('Predicción del alcance utilizando Holt-Winters con Validación Cruzada')
        plt.legend()
        fig2, ax2 = plt.gcf(), plt.gca() 
        st.pyplot(fig2, ax2, use_container_width=False) 

        
    with col1:
        st.markdown("<h2  style='font-size:25px;'><strong>VALIDACIÓN CRUZADA SARIMA FACEBOOK</strong></h2>", unsafe_allow_html=True)
        serie_temporal = datosMetricas['Alcance_Facebook']

        # Definir el número de pliegues para la validación cruzada k-fold
        n_splits = 5

        # Inicializar el objeto TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Inicializar una lista para almacenar los errores cuadráticos medios (MSE) de cada pliegue
        mse_scores = []

        # Definir una paleta de colores para los pliegues
        colors = plt.cm.viridis(np.linspace(0, 1, n_splits))

        for i, (train_index, test_index) in enumerate(tscv.split(serie_temporal), 1):
            # Dividir los datos en conjuntos de entrenamiento y prueba
            train, test = serie_temporal.iloc[train_index], serie_temporal.iloc[test_index]

            # Crear y ajustar el modelo SARIMA
            modelo_sarima = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 12))
            modelo_sarima_entrenado = modelo_sarima.fit()

            # Realizar predicciones en el conjunto de prueba
            predicciones = modelo_sarima_entrenado.get_forecast(steps=len(test)).predicted_mean

            # Calcular el error cuadrático medio (MSE) y agregarlo a la lista de puntuaciones MSE
            mse = mean_squared_error(test, predicciones)
            mse_scores.append(mse)

            st.write(f"Fold {i}: Error Cuadrático Medio (MSE): {mse}")

            # Visualizar resultados
            if i == 1:
                fig, ax = plt.subplots(figsize=(3, 1))  # Definir la figura y los ejes solo en el primer bucle
            ax.plot(test.index, predicciones, label=f'Fold {i} - Predicciones', color=colors[i-1], linestyle='-', linewidth=0.5)
        
        average_mse = np.mean(mse_scores)
        st.write("Error cuadrático medio promedio:", average_mse)

        # Mostrar la gráfica
        plt.plot(serie_temporal.index, serie_temporal, label='Serie Temporal', color='gray', linestyle='-', linewidth=0.5)
        ax.set_title('Predicciones del alcance utilizando SARIMA con Validación Cruzada')
        ax.set_xlabel('Total de publicaciones')
        ax.set_ylabel('Alcance total')
        ax.legend()
        st.pyplot(fig, use_container_width=False)
       

    with col2:
        st.markdown("<h2 style='font-size:25px;'><strong>VALIDACIÓN CRUZADA HOLT WINTERS FACEBOOK</strong></h2>", unsafe_allow_html=True)
        # Seleccionar la columna para la predicción
        serie_temporal = datosMetricas['Alcance_Facebook']

        # Definir el número de pliegues para la validación cruzada k-fold
        n_splits = 5

        # Inicializar el objeto TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Inicializar una lista para almacenar los errores cuadráticos medios (MSE) de cada pliegue
        mse_scores = []

        # Inicializar la figura para combinar todas las gráficas de los pliegues
        plt.figure(figsize=(3, 1))

        # Inicializar listas para almacenar predicciones y errores de cada fold
        all_predictions = []
        all_errors = []

        # Iterar sobre los pliegues generados por TimeSeriesSplit
        for fold_index, (train_index, test_index) in enumerate(tscv.split(serie_temporal)):
            # Dividir los datos en conjuntos de entrenamiento y prueba
            train, test = serie_temporal.iloc[train_index], serie_temporal.iloc[test_index]

            # Crear y ajustar el modelo Holt-Winters Exponential Smoothing
            modelo_hw = ExponentialSmoothing(train, seasonal='add', seasonal_periods=6)
            modelo_hw_entrenado = modelo_hw.fit()

            # Realizar predicciones en el conjunto de prueba
            predicciones = modelo_hw_entrenado.forecast(len(test))

            # Calcular el error cuadrático medio (MSE) y agregarlo a la lista de puntuaciones MSE
            mse = mean_squared_error(test, predicciones)
            mse_scores.append(mse)

            # Almacenar predicciones y errores
            all_predictions.append(predicciones)
            all_errors.append(mse)

            # Agregar la gráfica del fold actual a la figura combinada
            plt.plot(test.index, predicciones, label=f'Fold {fold_index + 1}', linewidth=0.5)

        # Mostrar predicciones y errores para cada fold
        for fold_index, (predicciones, error) in enumerate(zip(all_predictions, all_errors), start=1):
            st.write(f"Fold {fold_index}: Error Cuadrático Medio (MSE): {error}")

        # Calcular el error cuadrático medio promedio (MSE) de todos los pliegues
        average_mse = np.mean(mse_scores)
        st.write("Error cuadrático medio promedio:", average_mse)

        # Visualizar conjunto de entrenamiento y prueba
        plt.plot(serie_temporal.index, serie_temporal, label='Serie Temporal', color='gray', linestyle='-', linewidth=0.5)
        plt.xlabel('Total de publicaciones')
        plt.ylabel('Alcance total')
        plt.title('Predicción del alcance utilizando Holt-Winters con Validación Cruzada')
        plt.legend()
        fig2, ax2 = plt.gcf(), plt.gca() 
        st.pyplot(fig2, ax2, use_container_width=False) 



# Función para cargar datos
@st.cache
def cargar_datos(archivos):
    dfs = []
    for archivo in archivos:
        dfs.append(archivos[archivo])
    return pd.concat(dfs, ignore_index=True)


def chat_gpt(prompt, datosProyectos, datosSemilleros, datosPaises, datosCiudades, datosEdad):
    # Reemplaza 'your_api_key' con tu propia API Key de OpenAI
    openai.api_key = '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'

    # Construye el prompt incluyendo los datos
    full_prompt = f"{prompt} Proyectos: {datosProyectos}, Semilleros: {datosSemilleros}, Paises: {datosPaises}, Ciudades: {datosCiudades}, Edad: {datosEdad}"
    
    # Realiza una solicitud al endpoint correcto de la API de ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Puedes elegir otro modelo si lo deseas
        messages=[
            {"role": "system", "content": "You are a friendly assistant:"},
            {"role": "user", "content": full_prompt}
        ]
    )

    # Obtiene la respuesta generada por GPT
    response_text = response.choices[0].message['content']
    return response_text


    # Obtiene la respuesta generada por GPT
    response_text = response.choices[0].text.strip()
    return response_text


def chatbot(datosProyectos, datosSemilleros, datosPaises, datosCiudades, datosEdad):
    import streamlit as st
    
    if datosProyectos is not None \
            and datosSemilleros is not None and datosPaises is not None and datosCiudades is not None and datosEdad is not None:
        df = cargar_datos({
                           "datosProyectos": datosProyectos, "datosSemilleros": datosSemilleros,
                           "datosPaises": datosPaises, "datosCiudades": datosCiudades, "datosEdad": datosEdad})

        if df is not None:
            st.title("Chatea Con el grupo GALASH")
            if query := st.text_input("Digita tu pregunta"):
                st.text("Tu pregunta: " + query)
                if query:
                    # Llamada a la función chat_gpt
                    response = chat_gpt(query, datosProyectos, datosSemilleros, datosPaises, datosCiudades, datosEdad)
                    st.text("Respuesta del bot: " + response)


# Configuración del menú
st.set_page_config(
    page_title="GALASH - Análisis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Logo en la barra del menú, redondeado
with st.sidebar:
    st.image("logo.png", width=200, use_column_width=True, output_format="PNG", caption='Grupo de Investigación GALASH')

# Título de la aplicación
st.markdown("<h1 style='font-size:30px;'>Modelo Para La Gestión Estratégica De Conocimiento A Partir De La Información Contenida En Las Redes Sociales Del Grupo De Investigación GALASH</h1>", unsafe_allow_html=True)
# Estilo CSS para botones
st.markdown("""
    <style>
        .stButton>button {
            width: 200px;
        }
    </style>
""", unsafe_allow_html=True)
datos = cargar_archivo()
datosInstagram = cargar_archivoInsta()
datosMetricas = cargar_archivoMetricas()
datosEdad = cargar_archivoEdad()
datosProyectos = cargar_archivoProyectos()
datosSemilleros = cargar_archivoSemillero()
datosPaises = cargar_archivoPaises()
datosCiudades = cargar_archivoCiudades()


#Menu
def inicio():
    # Imagen de bienvenida
    from PIL import Image
    image = Image.open("images/inicio.jpeg")
    st.markdown("<h1 style='text-align: center;'>¡Bienvenidos!</h1>", unsafe_allow_html=True)
    st.write("Esta es una aplicación para explorar los datos, métricas y modelos de las redes sociales del grupo de investigación GALASH de manera interactiva.")

    st.image(image, use_column_width=True)

    # Descripción de la aplicación
    st.write("""
    ### ¿Qué puedes hacer aquí?
    - Explorar y visualizar datos.
    - Visualizar un análisis exploratorio de las redes sociales.
    - Visualizar predicciones de las redes sociales.
    - Interactuar con un chatbot para conocer más a fondo el grupo de investigación.
    """)

    st.write("--------")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("Proyecto de grado realizado por Andrea Salcedo y Edwin Sanabria")

def facebook():
    st.subheader("Facebook")

def instagram():
    st.subheader("Instagram")

def metricas():
    st.write("Página de Métricas")

def facebook():
    st.subheader("Facebook")

def instagram():
    st.subheader("Instagram")

def metricas():
    st.write("Página de Métricas")


def main():
    st.sidebar.title("Menú")
    menu_options = ["Inicio", "Facebook", "Instagram", "Metricas", "ChatBot"]
    choice = st.sidebar.selectbox("Seleccione una opción", menu_options)

    if choice == "Inicio":
        inicio()
    elif choice == "Facebook":
        facebook_option = st.sidebar.selectbox("Seleccione una opción de Facebook", ["Ver datos", "Analisis Exploratorio", "SARIMA - Holt Winters", "Validación"])
        if facebook_option == "Ver datos":
            if datos is not None:
                datosFacebook(datos)
            pass
        elif facebook_option == "Analisis Exploratorio":
            if datos is not None:
                mostrar_analisis(datos)
            pass
        elif facebook_option == "SARIMA - Holt Winters":
            if datos is not None:
                prediccionesFacebook(datos)
            pass
        elif facebook_option == "Validación":
            st.markdown("<p style='text-align: justify;'> La validación cruzada que se presenta a continuacion para el conjunto de datos de facebook es una técnica crucial para verificar la eficacia de métodos como SARIMA y Holt-Winters en el análisis de series temporales. Al emplear 5 folds en la validación cruzada, se divide el conjunto de datos en cinco partes, entrenando el modelo y evaluándolo, repitiendo este proceso cinco veces.</p>", unsafe_allow_html=True)
    
            if datos is not None:
                validacionFacebook(datos)
            pass
    elif choice == "Instagram":
        instagram_option = st.sidebar.selectbox("Seleccione una opción de Instagram", ["Ver datos", "Analisis Exploratorio", "SARIMA - Holt Winters", "Validación"])
        if instagram_option == "Ver datos":
            if datosInstagram is not None:
                datosInsta(datosInstagram)
            pass
        elif instagram_option == "Analisis Exploratorio":
            if datosInstagram is not None:
                analisis_instagram(datosInstagram)
            pass
        elif instagram_option == "SARIMA - Holt Winters":
            if datosInstagram is not None:
                prediccionesInstagram(datosInstagram)
            pass
        elif instagram_option == "Validación":
            st.markdown("<p style='text-align: justify;'> La validación cruzada que se presenta a continuacion para el conjunto de datos de instagram es una técnica crucial para verificar la eficacia de métodos como SARIMA y Holt-Winters en el análisis de series temporales. Al emplear 5 folds en la validación cruzada, se divide el conjunto de datos en cinco partes, entrenando el modelo y evaluándolo, repitiendo este proceso cinco veces.</p>", unsafe_allow_html=True)

            if datosInstagram is not None:
                validacionInstagram(datosInstagram)
            pass
    elif choice == "Metricas":
        metricas_option = st.sidebar.selectbox("Seleccione una opción de Metricas", ["Ver datos", "Analisis Exploratorio", "SARIMA - Holt Winters", "Validación"])
        if metricas_option == "Ver datos":
            if datosMetricas is not None:
                dataMetricas(datosMetricas)
            pass
        elif metricas_option == "Analisis Exploratorio":
            if datosMetricas is not None:
                analisis_metricas(datosMetricas)
            pass
        elif metricas_option == "SARIMA - Holt Winters":
            if datosMetricas is not None:
                prediccionesMetricas(datosMetricas)
            pass
        elif metricas_option == "Validación":
            st.markdown("<p style='text-align: justify;'> La validación cruzada que se presenta a continuacion para facebook e instagram es una técnica crucial para verificar la eficacia de métodos como SARIMA y Holt-Winters en el análisis de series temporales. Al emplear 5 folds en la validación cruzada, se divide el conjunto de datos en cinco partes, entrenando el modelo y evaluándolo, repitiendo este proceso cinco veces.</p>", unsafe_allow_html=True)

            if datosMetricas is not None:
                validacionMetricas(datosMetricas)
            pass
    elif choice == "ChatBot":
        chatbot(datosProyectos, datosSemilleros, datosPaises, datosCiudades, datosEdad)

if __name__ == "__main__":
    main()