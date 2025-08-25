import pandas as pd
import joblib
from modelo import ModeloMultietiqueta
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Clase para el modelo multi-etiqueta
class ModeloMultietiqueta:
    def __init__(self, modelo, categorias):
        self.modelo = modelo
        self.categorias = categorias

# Cargar datos
df = pd.read_csv('data/raw/DatabaseBio.csv', sep=';')

# Separar features y etiquetas
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

# Guardar modelo completo con vectorizer y categorías
modelo_final = ModeloMultietiqueta(modelo=multi_rf, categorias=categorias)
joblib.dump({'modelo': modelo_final, 'vectorizer': vectorizer}, 'modelos/modelo_entrenado.pkl')

print("Modelo entrenado y guardado correctamente.")
