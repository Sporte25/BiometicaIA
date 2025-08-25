# src/train.py
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# ----------------------------
# Funciones
# ----------------------------

def cargar_datos(ruta_csv):
    # Cargar CSV separado por ;
    df = pd.read_csv(ruta_csv, sep=';')
    
    # Convertir grupos a listas
    df['group_list'] = df['group'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    
    return df

def preparar_datos(df):
    # Texto combinado: title + abstract
    df['texto_combinado'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    
    X = df['texto_combinado'].values
    
    # Binarizar etiquetas
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['group_list'])
    
    return X, y, mlb

def entrenar_modelo(X_train, y_train):
    # Vectorizador TF-IDF
    vectorizador = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    
    # Random Forest multietiqueta
    modelo = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
    
    # Pipeline vectorizador + clasificador
    X_train_tfidf = vectorizador.fit_transform(X_train)
    modelo.fit(X_train_tfidf, y_train)
    
    return modelo, vectorizador

def evaluar_modelo(modelo, vectorizador, X_test, y_test, mlb):
    X_test_tfidf = vectorizador.transform(X_test)
    y_pred = modelo.predict(X_test_tfidf)
    
    print("F1 Score (Weighted):", f1_score(y_test, y_pred, average='weighted', zero_division=0))
    print("F1 Score (Macro):", f1_score(y_test, y_pred, average='macro', zero_division=0))
    print("F1 Score (Micro):", f1_score(y_test, y_pred, average='micro', zero_division=0))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta CSV de datos")
    parser.add_argument("--output", required=True, help="Ruta para guardar modelo entrenado")
    args = parser.parse_args()
    
    # Crear carpeta de modelos si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Cargar datos
    df = cargar_datos(args.input)
    print(f"Datos cargados: {len(df)} registros")
    
    # Preparar datos
    X, y, mlb = preparar_datos(df)
    
    # Train/test split con stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split realizado: {len(X_train)} train, {len(X_test)} test")
    
    # Entrenar modelo
    print("Entrenando modelo OneVsRest con Random Forest...")
    modelo, vectorizador = entrenar_modelo(X_train, y_train)
    
    # Guardar modelo y vectorizador
    joblib.dump({'modelo': modelo, 'vectorizador': vectorizador, 'mlb': mlb}, args.output)
    print(f"Modelo guardado en: {args.output}")
    
    # Evaluar modelo
    print("Evaluando modelo en conjunto de test...")
    evaluar_modelo(modelo, vectorizador, X_test, y_test, mlb)
