import pandas as pd
from modelo import ModeloMultietiqueta
import joblib

class ClasificadorArticulosMedicos:
    def __init__(self, ruta_modelo):
        data = joblib.load(ruta_modelo)
        self.modelo = data['modelo']  # Contiene ModeloMultietiqueta
        self.vectorizer = data['vectorizer']
        self.categorias = self.modelo.categorias

    def predecir(self, df):
        # Combinar título y abstract
        df_proc = df.copy()
        df_proc['texto_combinado'] = df_proc['title'].astype(str) + ' ' + df_proc['abstract'].astype(str)

        # Vectorizar
        X_vect = self.vectorizer.transform(df_proc['texto_combinado'])

        # Predecir
        pred_matrix = self.modelo.modelo.predict(X_vect)  # RandomForestMultiOutputClassifier
        df_pred = pd.DataFrame(pred_matrix, columns=self.categorias)

        # Crear columna con etiquetas predichas
        df_proc['group_predicted'] = df_pred.apply(lambda row: [cat for cat in self.categorias if row[cat]==1], axis=1)
        return df_proc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=';')
    clasificador = ClasificadorArticulosMedicos(args.model)
    df_resultado = clasificador.predecir(df)
    df_resultado.to_csv(args.output, index=False)
    print("Predicciones guardadas en:", args.output)
