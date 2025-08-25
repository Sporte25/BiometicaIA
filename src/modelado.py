import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report

class ModeloMultietiqueta:
    def __init__(self):
        self.vectorizador = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.modelo = None

    def entrenar_modelo_ovr(self, X, y, tipo_modelo='logistic', guardar=False, nombre_archivo='modelo_entrenado.pkl'):
        """Entrenamiento con OneVsRest"""
        if tipo_modelo == 'logistic':
            clasificador = LogisticRegression(max_iter=1000, random_state=42)
        elif tipo_modelo == 'svm':
            clasificador = LinearSVC(random_state=42)
        elif tipo_modelo == 'random_forest':
            clasificador = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Tipo de modelo '{tipo_modelo}' no soportado.")

        self.modelo = Pipeline([
            ('tfidf', self.vectorizador),
            ('clf', OneVsRestClassifier(clasificador))
        ])

        self.modelo.fit(X, y)

        if guardar:
            self.guardar_modelo(nombre_archivo)

        return self.modelo

    def entrenar_binary_relevance(self, X, y, tipo_modelo='logistic', guardar=False, nombre_archivo='modelo_br.pkl'):
        """Entrenamiento con Binary Relevance"""
        if tipo_modelo == 'logistic':
            clasificador = LogisticRegression(max_iter=1000, random_state=42)
        elif tipo_modelo == 'svm':
            clasificador = LinearSVC(random_state=42)
        else:
            raise ValueError(f"Tipo de modelo '{tipo_modelo}' no soportado.")

        pipeline = Pipeline([
            ('tfidf', self.vectorizador),
            ('clf', BinaryRelevance(clasificador))
        ])

        pipeline.fit(X, y)
        self.modelo = pipeline

        if guardar:
            self.guardar_modelo(nombre_archivo)

        return pipeline

    def predecir(self, X):
        if self.modelo is None:
            raise ValueError("No hay modelo entrenado para realizar predicciones.")
        return self.modelo.predict(X)

    def evaluar(self, X, y, categorias=None):
        predicciones = self.predecir(X)
        print("F1 Score (weighted):", f1_score(y, predicciones, average='weighted'))
        print("\nReporte de Clasificación:")
        if categorias:
            print(classification_report(y, predicciones, target_names=categorias))
        else:
            print(classification_report(y, predicciones))
        return predicciones

    def guardar_modelo(self, nombre_archivo='modelo_entrenado_2.pkl'):
        """Guarda el modelo entrenado en la carpeta 'modelos'"""
        if self.modelo is None:
            raise ValueError("No hay modelo entrenado para guardar.")

        ruta_carpeta = os.path.join(os.path.dirname(__file__), '..', 'modelos')
        os.makedirs(ruta_carpeta, exist_ok=True)

        ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
        joblib.dump(self.modelo, ruta_completa)
        print(f"Modelo guardado en: {ruta_completa}")