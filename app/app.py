# app/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("PYTHONPATH:", sys.path)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from src.predict import ClasificadorArticulosMedicos
from src.evaluacion import evaluar_modelo_multietiqueta
import joblib
import matplotlib.pyplot as plt
import sys
import os

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Artículos Médicos",
    page_icon="🏥",
    layout="wide"
)

# Título de la aplicación
st.title("🏥 Clasificador de Artículos Médicos")
st.markdown("Sistema de IA para clasificar literatura médica en dominios específicos")

# Sidebar para carga de archivos y configuración
st.sidebar.header("Configuración")
archivo_modelo = st.sidebar.file_uploader("Cargar modelo entrenado", type=["pkl", "joblib"])
archivo_datos = st.sidebar.file_uploader("Cargar datos para clasificar", type=["csv"])

# Si se cargó un archivo de datos
if archivo_datos is not None:
    # Cargar datos
    df = pd.read_csv(archivo_datos)
    
    # Mostrar datos
    st.subheader("Datos Cargados")
    st.dataframe(df.head())
    
    # Si también se cargó un modelo
    if archivo_modelo is not None:
        # Guardar modelo temporalmente
        with open("modelo_temporal.pkl", "wb") as f:
            f.write(archivo_modelo.getvalue())
        
        # Inicializar clasificador
        clasificador = ClasificadorArticulosMedicos("modelo_temporal.pkl")
        
        # Realizar predicciones
        with st.spinner("Clasificando artículos..."):
            df_resultado = clasificador.predecir(df)
        
        # Mostrar resultados
        st.subheader("Resultados de la Clasificación")
        st.dataframe(df_resultado[['title', 'abstract', 'group_predicted']])
        
        # Botón para descargar resultados
        st.download_button(
            label="Descargar resultados CSV",
            data=df_resultado.to_csv(index=False),
            file_name="resultados_clasificacion.csv",
            mime="text/csv"
        )
        
        # Si el dataset original tenía etiquetas, mostrar evaluación
        if 'group' in df.columns:
            st.subheader("Evaluación del Modelo")
            
            # Evaluar desempeño
            _, resultados = clasificador.evaluar_desempeno(df)
            
            # Mostrar métricas principales
            col1, col2, col3 = st.columns(3)
            col1.metric("F1 Score (Weighted)", f"{resultados['f1_weighted']:.4f}")
            col2.metric("F1 Score (Macro)", f"{resultados['f1_macro']:.4f}")
            col3.metric("F1 Score (Micro)", f"{resultados['f1_micro']:.4f}")
            
            # Mostrar matriz de confusión
            st.subheader("Matrices de Confusión")
            
            # Calcular matrices de confusión por categoría
            from ast import literal_eval
            categorias = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
            
            y_true = []
            for grupo in df['group']:
                if isinstance(grupo, str):
                    grupo = literal_eval(grupo)
                vector = [1 if cat in grupo else 0 for cat in categorias]
                y_true.append(vector)
            
            y_true = np.array(y_true)
            
            y_pred = []
            for grupo in df_resultado['group_predicted']:
                vector = [1 if cat in grupo else 0 for cat in categorias]
                y_pred.append(vector)
            
            y_pred = np.array(y_pred)
            
            # Crear visualización de matrices de confusión
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.ravel()
            
            for i, categoria in enumerate(categorias):
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true[:, i], y_pred[:, i])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(categoria)
                axes[i].set_xlabel('Predicho')
                axes[i].set_ylabel('Real')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        st.warning("Por favor, carga un modelo entrenado para realizar clasificaciones.")

else:
    st.info("Por favor, carga un archivo CSV con los datos para clasificar. El archivo debe contener las columnas 'title' y 'abstract'.")

# Sección de información
st.sidebar.markdown("---")
st.sidebar.info(
    """
    Esta aplicación clasifica artículos médicos en cuatro dominios:
    - Cardiovascular
    - Neurológico
    - Hepatorenal
    - Oncológico
    
    Desarrollado para el Challenge de Clasificación Biomédica con IA.
    """
)