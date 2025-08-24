# src/predict.py
import pandas as pd
import numpy as np
import joblib
from preprocesamiento import PreprocesadorTexto, combinar_textos


class ClasificadorArticulosMedicos:
    def __init__(self, ruta_modelo=None):
        self.preprocesador = PreprocesadorTexto()
        self.modelo = None
        self.categorias = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
        
        if ruta_modelo:
            self.cargar_modelo(ruta_modelo)
    
    def cargar_modelo(self, ruta_modelo):
        """Cargar modelo entrenado"""
        data = joblib.load(ruta_modelo)

        # Si es un diccionario, extraemos el modelo y categorías
        if isinstance(data, dict):
            self.modelo = data["modelo"]
            if "categorias" in data:
                self.categorias = data["categorias"]
        else:
            self.modelo = data
    
    def preprocesar_datos(self, df):
        """Preprocesar datos nuevos"""
        df_procesado = df.copy()
        
        # Combinar título y abstract
        df_procesado = combinar_textos(df_procesado)
        
        # Preprocesar texto
        df_procesado = self.preprocesador.preprocesar_dataframe(
            df_procesado, 
            ['texto_combinado'], 
            usar_spacy=False
        )
        
        return df_procesado
    
    def predecir(self, df):
        """Realizar predicciones sobre nuevos datos"""
        # Preprocesar
        df_procesado = self.preprocesar_datos(df)
        
        # Predecir
        X = df_procesado['texto_combinado']
        predicciones = self.modelo.predict(X)
        
        # Convertir a formato de lista
        predicciones_lista = []
        for pred in predicciones:
            etiquetas = []
            for i, valor in enumerate(pred):
                if valor == 1:
                    etiquetas.append(self.categorias[i])
            predicciones_lista.append(etiquetas)
        
        # Añadir al DataFrame
        df['group_predicted'] = predicciones_lista
        
        return df
    
    def evaluar_desempeno(self, df):
        """Evaluar el desempeño si tenemos las etiquetas reales"""
        if 'group' not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'group' para evaluación")
        
        # Predecir
        df_con_predicciones = self.predecir(df)
        
        # Convertir group a formato binario para evaluación
        from ast import literal_eval
        y_true = []
        for grupo in df['group']:
            if isinstance(grupo, str):
                grupo = literal_eval(grupo)
            vector = [1 if cat in grupo else 0 for cat in self.categorias]
            y_true.append(vector)
        
        y_true = np.array(y_true)
        
        # Convertir predicciones a formato binario
        y_pred = []
        for grupo in df_con_predicciones['group_predicted']:
            vector = [1 if cat in grupo else 0 for cat in self.categorias]
            y_pred.append(vector)
        
        y_pred = np.array(y_pred)
        
        # Evaluar
        from evaluacion import evaluar_modelo_multietiqueta
        resultados = evaluar_modelo_multietiqueta(y_true, y_pred, self.categorias)
        
        return df_con_predicciones, resultados


# Función principal para ejecutar desde línea de comandos
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Clasificar artículos médicos')
    parser.add_argument('--input', required=True, help='Ruta al archivo CSV de entrada')
    parser.add_argument('--output', required=True, help='Ruta al archivo CSV de salida')
    parser.add_argument('--model', required=True, help='Ruta al modelo entrenado')
    parser.add_argument('--evaluate', action='store_true', help='Si se incluye, evalúa y guarda matriz de confusión')

    args = parser.parse_args()
    
    # Cargar datos
    df = pd.read_csv(args.input)
    
    # Clasificador
    clasificador = ClasificadorArticulosMedicos(args.model)
    
    if args.evaluate:
        print("🔍 Ejecutando predicción + evaluación...")
        df_resultado, resultados = clasificador.evaluar_desempeno(df)
        df_resultado.to_csv(args.output, index=False)
        print(f"✅ Resultados + métricas guardados en: {args.output}")
    else:
        print("🤖 Ejecutando solo predicción...")
        resultado = clasificador.predecir(df)
        resultado.to_csv(args.output, index=False)
        print(f"✅ Predicciones guardadas en: {args.output}")


if __name__ == "__main__":
    main()
