import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff


# Función para cargar datos desde un archivo
def cargar_datos():
    archivo_cargado = st.file_uploader("Selecciona un archivo", type=["csv", "xlsx"])

    if archivo_cargado is not None:
        try:
            if archivo_cargado.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                dataset = pd.read_excel(archivo_cargado)
            else:
                dataset = pd.read_csv(archivo_cargado)
            return dataset
        except Exception as e:
            st.error(f"Error: {e}")
    return None

# Función para verificar si una variable numérica es continua o discreta
def verificar_continua_o_discreta(columna):
    if columna.nunique() > 50 or columna.dtype == 'float64':
        return 'continua'
    else:
        return'discreta'


# Función para mostrar la pantalla de bienvenida
def pantalla_bienvenida():
    st.title("Bienvenido a Proyecto Uno")
    st.subheader("Integrantes:")
    st.write("Alejandra Sierra - 16002653")
    st.write("Cindy Gutierrez - 20016132")
    st.write("Edgar Cristal - 22007686")
    st.write("Por favor, carga un archivo CSV o Excel para comenzar.")
    dataset = cargar_datos()
    return dataset


# Función para mostrar estadísticas resumidas y gráficos
def analisis_de_datos(dataset):
    st.title("Análisis de Datos")

    st.subheader("Lista de Variables:")
    st.write("Variables Numéricas Continuas:")
    
    # Aplicar la verificación a cada columna numérica
    for columna in dataset.select_dtypes(include=['float64', 'int64']).columns:
        clasificacion = verificar_continua_o_discreta(dataset[columna])
        if clasificacion == 'continua':
            st.write(columna)

    st.write("Variables Numéricas Discretas:")
    
    # Aplicar la verificación a cada columna numérica
    for columna in dataset.select_dtypes(include='int64').columns:
        clasificacion = verificar_continua_o_discreta(dataset[columna])
        if clasificacion == 'discreta':
            st.write(columna)

            # Estadísticas y gráfico para variables numéricas discretas
            st.subheader(f"Análisis de {columna}")
            st.write("Estadísticas:")
            st.write(f"Media: {dataset[columna].mean()}")
            st.write(f"Mediana: {dataset[columna].median()}")
            st.write(f"Desviación Estándar: {dataset[columna].std()}")
            st.write(f"Varianza: {dataset[columna].var()}")
            st.write(f"Moda: {dataset[columna].mode().values[0]}")

    # Gráfico de histograma interactivo con Plotly Express
            fig_hist = px.histogram(dataset, x=columna, nbins=30, title=f"Histograma de {columna}")
            fig_hist.update_layout(
                xaxis_title=columna,
                yaxis_title='Frecuencia',
                bargap=0.05
            )
            st.plotly_chart(fig_hist)


    # Selector de variable para el gráfico KDE
    variable = st.selectbox("Variable", ['Age', 'Quantity'])

    # Mínimo y máximo de la variable seleccionada
    min_value = dataset[variable].min()
    max_value = dataset[variable].max()

    # Crear gráfico KDE interactivo con Streamlit
    st.subheader(f"Densidad de {variable}")
    rangos = st.slider(f"Valor de {variable}", min_value, max_value, (min_value, max_value))
    r_min = rangos[0]
    r_max = rangos[1]

    kde = stats.gaussian_kde(dataset[variable])
    x = np.linspace(r_min, r_max, 1000)
    y = kde.evaluate(x)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(dataset[variable], color="blue", shade=False)
    ax.vlines(x=r_min, ymin=0, ymax=kde.evaluate(r_min), color="blue")
    ax.vlines(x=r_max, ymin=0, ymax=kde.evaluate(r_max))
    ax.fill_between(x, y, color="orange", alpha=0.5)
    plt.title(f"Densidad de {variable}")
    st.pyplot(fig)
    st.text(f"Probabilidad: {np.round(np.sum(y), 4)/100}")

    # Identificar columnas categóricas y continuas
    categoricas = [col for col in dataset.columns if dataset[col].dtype == 'object']
    continuas = [col for col in dataset.columns if dataset[col].dtype in ['float64', 'int64']]

    # Selector de variables para el gráfico de caja
    variableA = st.selectbox("Variable Continua", continuas)
    variableB = st.selectbox("Variable Discreta", categoricas)

    # Crear gráfico de caja interactivo con Streamlit
    st.subheader("Gráfico de Caja")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=dataset, x=variableB, y=variableA)
    st.pyplot(fig2)

#### finaliza analisis

def main():
    st.set_page_config(page_title="Aplicación de Análisis de Datos", page_icon=":bar_chart:")

    # Crear pestañas
    pestañas = ["Bienvenida", "Análisis de Datos"]
    pestaña_actual = st.sidebar.radio("Selecciona Pestaña", pestañas)

    if pestaña_actual == "Bienvenida":
        dataset = pantalla_bienvenida()
        if dataset is not None:
            st.subheader("Vista previa del conjunto de datos:")
            st.dataframe(dataset.head())

    elif pestaña_actual == "Análisis de Datos":
        st.title("Análisis de Datos")
        dataset = cargar_datos()
        if dataset is not None:
            analisis_de_datos(dataset)

if __name__ == '__main__':
    main()
