import pandas as pd
import joblib
from modelo import ModeloMultietiqueta
from sklearn.metrics import classification_report, confusion_matrix

class ModeloMultietiqueta:
    def __init__(self, modelo, categorias):
        self.modelo = modelo
        self.categorias = categorias

class ClasificadorArticulosMedicos:
    def __init__(self, ruta_modelo):
        data = joblib.load(ruta_modelo)
        self.modelo = data['modelo']
        self.vectorizer = data['vectorizer']
        self.categorias = self.modelo.categorias

    def predecir(self, df):
        df_proc = df.copy()
        df_proc['texto_combinado'] = df_proc['title'].astype(str) + ' ' + df_proc['abstract'].astype(str)
        X_vect = self.vectorizer.transform(df_proc['texto_combinado'])
        pred_matrix = self.modelo.modelo.predict(X_vect)
        df_pred = pd.DataFrame(pred_matrix, columns=self.categorias)
        df_proc['group_predicted'] = df_pred.apply(lambda row: [cat for cat in self.categorias if row[cat] == 1], axis=1)
        return df_proc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--true-labels', required=False)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=';')
    clasificador = ClasificadorArticulosMedicos(args.model)
    df_resultado = clasificador.predecir(df)

    if args.true_labels:
        df_true = pd.read_csv(args.true_labels, sep=';')
        df_resultado['group_true'] = df_true['group'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else '')
        df_resultado['group_predicted'] = df_resultado['group_predicted'].apply(lambda x: x[0] if x else '')

        print("📈 Classification Report:")
        print(classification_report(df_resultado['group_true'], df_resultado['group_predicted']))

        print("🧮 Confusion Matrix:")
        print(confusion_matrix(df_resultado['group_true'], df_resultado['group_predicted']))

    df_resultado.to_csv(args.output, index=False)
    print("✅ Predictions saved to:", args.output)