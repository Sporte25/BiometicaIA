# src/generar_datos.py
import pandas as pd
import os

def generar_dataset():
    # Crear carpeta data si no existe
    os.makedirs("data", exist_ok=True)

    # Datos de ejemplo
    data = {
        "title": [
            "Advances in Cardiovascular Surgery",
            "Neurological Outcomes after Stroke",
            "Liver and Kidney Function in Critical Care",
            "Recent Trends in Oncology Research",
            "Multidisciplinary Approaches in Medicine"
        ],
        "abstract": [
            "This paper discusses new methods in heart surgery with reduced risks.",
            "Stroke patients show improved recovery using new therapies.",
            "Critical care patients often face complications in liver and kidney.",
            "Oncology research highlights new immunotherapy drugs.",
            "Integrative approaches across specialties improve patient outcomes."
        ],
        "group": [
            "['Cardiovascular']",
            "['Neurological']",
            "['Hepatorenal']",
            "['Oncological']",
            "['Cardiovascular','Oncological','Neurological']"
        ]
    }

    # Crear DataFrame
    df = pd.DataFrame(data)

    # Guardar en data/articulos.csv
    ruta = "data/articulos.csv"
    df.to_csv(ruta, index=False, encoding="utf-8")

    print(f"Archivo de ejemplo creado en: {ruta}")

if __name__ == "__main__":
    generar_dataset()
