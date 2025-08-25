import sys
from pathlib import Path

# Añade la carpeta src al path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import pandas as pd
import streamlit as st
from predict import ClasificadorArticulosMedicos, ModeloMultietiqueta
#from src.predict import ClasificadorArticulosMedicos, ModeloMultietiqueta


st.title("Clasificación de Artículos Médicos")

uploaded_file = st.file_uploader("Sube un CSV con columnas: title y abstract", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        st.write("Datos cargados correctamente:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error cargando datos: {e}")

    if st.button("Cargar Modelo y Predecir"):
        try:
            clasificador = ClasificadorArticulosMedicos('modelos/modelo_entrenado.pkl')
            df_resultado = clasificador.predecir(df)
            st.write("Predicciones:")
            st.dataframe(df_resultado[['title', 'group_predicted']])
        except Exception as e:
            st.error(f"Error cargando modelo o prediciendo: {e}")
