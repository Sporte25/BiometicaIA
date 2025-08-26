import pandas as pd
import joblib
from modelo import ModeloMultietiqueta
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Clase para el modelo multi-etiqueta
class ModeloMultietiqueta:
    def __init__(self, modelo, categorias):
        self.modelo = modelo
        self.categorias = categorias

# Cargar datos
df = pd.read_csv('data/raw/DatabaseBio.csv', sep=';')
df['group'] = df['group'].fillna('')
X = df['title'].astype(str) + ' ' + df['abstract'].astype(str)
y = df['group'].str.get_dummies(sep='|')
categorias = list(y.columns)

# Vectorización
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Modelo Random Forest multi-etiqueta
rf = RandomForestClassifier(n_estimators=100, random_state=42)
multi_rf = MultiOutputClassifier(rf)
multi_rf.fit(X_train, y_train)

# Evaluación
y_pred = multi_rf.predict(X_test)
print("📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=categorias))

# Visualización de características importantes
def plot_feature_importance(vectorizer, modelo):
    importances = modelo.estimators_[0].feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    indices = np.argsort(importances)[-20:]

    plt.figure(figsize=(10, 6))
    plt.title("Top 20 Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("modelos/feature_importance.png")
    plt.close()

plot_feature_importance(vectorizer, multi_rf)

# Guardar modelo
modelo_final = ModeloMultietiqueta(modelo=multi_rf, categorias=categorias)
joblib.dump({'modelo': modelo_final, 'vectorizer': vectorizer}, 'modelos/modelo_entrenado.pkl')
print("✅ Model trained and saved successfully.")