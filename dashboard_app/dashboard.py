import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from modelo import ModeloMultietiqueta
from sklearn.metrics import classification_report, confusion_matrix
from predict import ClasificadorArticulosMedicos

st.set_page_config(page_title="Medical Article Classifier", layout="wide")
st.title("Medical Article Classification")

from sklearn.preprocessing import MultiLabelBinarizer

def show_metrics(df):
    if 'group_true' in df.columns:
        # Asegurar que ambas columnas estén en formato lista
        true_labels = df['group_true'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
        pred_labels = df['group_predicted'].apply(lambda x: x if isinstance(x, list) else [])

        mlb = MultiLabelBinarizer()
        y_true_bin = mlb.fit_transform(true_labels)
        y_pred_bin = mlb.transform(pred_labels)

        report = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, output_dict=True)
        st.write("Evaluation Metrics")
        st.dataframe(pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score', 'support']])
    else:
        st.info("No true labels ('group_true') found to compute metrics.")

def show_confusion_matrix(df):
    if 'group_true' in df.columns:
        y_true = df['group_true'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else '')
        y_pred = df['group_predicted'].apply(lambda x: x[0] if isinstance(x, list) and x else '')
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

def show_class_distribution(df):
    fig, ax = plt.subplots()
    df['group_predicted'].apply(lambda x: x[0] if isinstance(x, list) and x else '').value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Predicted Class Distribution")
    st.pyplot(fig)

def show_feature_importance():
    import os
    from PIL import Image
    path = 'modelos/feature_importance.png'
    if os.path.exists(path):
        st.image(Image.open(path), caption="Top 20 Feature Importances", use_column_width=True)
    else:
        st.info("Feature importance visualization not available.")

@st.cache_resource
def load_model():
    return ClasificadorArticulosMedicos('modelos/modelo_entrenado.pkl')

model = load_model()

tab1, tab2 = st.tabs(["Predict from file", "Manual demo"])

with tab1:
    uploaded_file = st.file_uploader("Upload a CSV with columns: title, abstract (and optionally group_true)", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=';')
            st.success("File loaded successfully")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")

        if st.button("🔍 Predict"):
            try:
                df_result = model.predecir(df)
                st.subheader("Predictions")
                st.dataframe(df_result[['title', 'group_predicted']])

                show_metrics(df_result)
                show_confusion_matrix(df_result)
                show_class_distribution(df_result)
                show_feature_importance()

            except Exception as e:
                st.error(f"Error during prediction: {e}")

with tab2:
    st.write("Try classifying a custom article:")
    title_demo = st.text_input("Article title")
    abstract_demo = st.text_area("Article abstract")

    if st.button("Classify article"):
        if title_demo.strip() and abstract_demo.strip():
            demo_df = pd.DataFrame([{'title': title_demo, 'abstract': abstract_demo}])
            try:
                demo_result = model.predecir(demo_df)
                st.success(f"Prediction: {demo_result.iloc[0]['group_predicted']}")
            except Exception as e:
                st.error(f"Error during manual prediction: {e}")
        else:
            st.warning("Please enter both title and abstract.")