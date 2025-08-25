import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from predict import ClasificadorArticulosMedicos, ModeloMultietiqueta

# General configuration
st.set_page_config(page_title="Medical Article Classifier", layout="wide")
st.title("🧠 Medical Article Classification")

# Function: show metrics
def show_metrics(df):
    if 'group_true' in df.columns:
        report = classification_report(df['group_true'], df['group_predicted'], output_dict=True)
        st.write("📈 Evaluation Metrics")
        st.dataframe(pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score', 'support']])
    else:
        st.info("No true labels ('group_true') found to compute metrics.")

# Function: confusion matrix
def show_confusion_matrix(df):
    if 'group_true' in df.columns:
        cm = confusion_matrix(df['group_true'], df['group_predicted'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# Function: class distribution
def show_class_distribution(df):
    fig, ax = plt.subplots()
    df['group_predicted'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("📌 Predicted Class Distribution")
    st.pyplot(fig)

# Function: feature importance
def show_feature_importance(clasificador):
    if hasattr(clasificador, 'mostrar_importancias'):
        fig = clasificador.mostrar_importancias()
        st.pyplot(fig)
    else:
        st.info("This model does not support feature importance visualization.")

# Load model once
@st.cache_resource
def load_model():
    return ClasificadorArticulosMedicos('modelos/modelo_entrenado.pkl')

model = load_model()

# Tabs for separate workflows
tab1, tab2 = st.tabs(["📁 Predict from file", "🧪 Manual demo"])

# TAB 1: Predict from file
with tab1:
    uploaded_file = st.file_uploader("Upload a CSV with columns: title, abstract (and optionally group_true)", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=';')
            st.success("✅ File loaded successfully")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

        if st.button("🔍 Predict"):
            try:
                df_result = model.predecir(df)
                st.subheader("📊 Predictions")
                st.dataframe(df_result[['title', 'group_predicted']])

                show_metrics(df_result)
                show_confusion_matrix(df_result)
                show_class_distribution(df_result)
                show_feature_importance(model)

            except Exception as e:
                st.error(f"❌ Error loading model or making predictions: {e}")

# TAB 2: Manual demo
with tab2:
    st.write("Try classifying a custom article:")
    title_demo = st.text_input("📝 Article title")
    abstract_demo = st.text_area("📄 Article abstract")

    if st.button("🚀 Classify article"):
        if title_demo.strip() and abstract_demo.strip():
            demo_df = pd.DataFrame([{'title': title_demo, 'abstract': abstract_demo}])
            try:
                demo_result = model.predecir(demo_df)
                st.success(f"🔎 Prediction: {demo_result.iloc[0]['group_predicted']}")
            except Exception as e:
                st.error(f"❌ Error during manual prediction: {e}")
        else:
            st.warning("Please enter both title and abstract.")