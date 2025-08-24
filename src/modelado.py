# src/modelado.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
    
    def evaluar(self, X, y):
        predicciones = self.predecir(X)
        print("F1 Score (weighted):", f1_score(y, predicciones, average='weighted'))
        print("\nReporte de Clasificación:")
        print(classification_report(y, predicciones, target_names=categorias))
        return predicciones