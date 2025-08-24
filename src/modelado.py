# src/modelado.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import joblib
import os
from ast import literal_eval


class ModeloMultietiqueta:
    def __init__(self):
        self.vectorizador = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.modelo = None

    def entrenar_modelo_ovr(self, X, y, tipo_modelo='logistic'):
        """OneVsRest approach"""
        if tipo_modelo == 'logistic':
            clasificador = LogisticRegression(max_iter=1000, random_state=42)
        elif tipo_modelo == 'svm':
            clasificador = LinearSVC(random_state=42)
        elif tipo_modelo == 'random_forest':
            clasificador = RandomForestClassifier(random_state=42)

        self.modelo = Pipeline([
            ('tfidf', self.vectorizador),
            ('clf', OneVsRestClassifier(clasificador))
        ])

        self.modelo.fit(X, y)
        return self.modelo

    def entrenar_binary_relevance(self, X, y, tipo_modelo='logistic'):
        """Binary Relevance approach"""
        if tipo_modelo == 'logistic':
            clasificador = LogisticRegression(max_iter=1000, random_state=42)
        elif tipo_modelo == 'svm':
            clasificador = LinearSVC(random_state=42)

        pipeline = Pipeline([
            ('tfidf', self.vectorizador),
            ('clf', BinaryRelevance(clasificador))
        ])

        pipeline.fit(X, y)
        return pipeline

    def predecir(self, X):
        return self.modelo.predict(X)

    def evaluar(self, X, y, categorias):
        predicciones = self.predecir(X)
        print("F1 Score (weighted):", f1_score(y, predicciones, average='weighted'))
        print("\nReporte de Clasificación:")
        print(classification_report(y, predicciones, target_names=categorias))
        return predicciones


def main():
    # 🔹 Cargar dataset de ejemplo
    df = pd.read_csv("data/articulos.csv")
    df["texto"] = df["title"] + " " + df["abstract"]

    # 🔹 Transformar etiquetas a multilabel binario
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["group"].apply(literal_eval))

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], y, test_size=0.2, random_state=42
    )

    # 🔹 Entrenar modelo OneVsRest con regresión logística
    modelo = ModeloMultietiqueta()
    modelo_entrenado = modelo.entrenar_modelo_ovr(X_train, y_train, tipo_modelo='logistic')

    # 🔹 Evaluar modelo en el test set
    print("\n=== Evaluación del modelo en test ===")
    modelo.evaluar(X_test, y_test, mlb.classes_)

    # 🔹 Guardar artefacto (modelo + binarizador)
    os.makedirs("modelos", exist_ok=True)
    artefacto = {
        "modelo": modelo_entrenado,
        "mlb": mlb
    }
    joblib.dump(artefacto, "modelos/modelo_entrenado.pkl")
    print("\n✅ Modelo entrenado y guardado en modelos/modelo_entrenado.pkl")


if __name__ == "__main__":
    main()
