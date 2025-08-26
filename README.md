# BiometicaIA — Medical Article Classifier

This project implements a multi-label classification system for biomedical articles using machine learning. It allows users to upload CSV files containing article titles and abstracts, and predicts relevant medical categories. The app includes interactive visualizations such as confusion matrix, class distribution, and feature importance.

## Features

- Upload CSV files with `title`, `abstract`, and optionally `group` (true labels)
- Predict medical categories using a trained Random Forest model
- View weighted F1-score, confusion matrix, and class distribution
- Classify custom articles manually
- Visualize top 20 most important features

## Execution
1. Train the model
  python train.py
- This will generate modelo_entrenado.pkl and feature_importance.png inside the modelos/ folder.

2. Predict from file
   python predict.py \
  --input data/ejemplos/ejemplo_1.csv \
  --output resultados.csv \
  --model modelos/modelo_entrenado.pkl \
  --true-labels data/ejemplos/ejemplo_1.csv

3. Launch the dashboard
   streamlit run dashboard_app/dashboard.py

## Input Format
- CSV file must contain:
- title;abstract;group
  - title: Title of the article
  - abstract: Abstract or summary
  - group: One or more medical categories (separated by | for multi-label)
- Example:
   title;abstract;group
   Cardiovascular risk in diabetic patients;A cohort study was performed...;cardiology|diabetes

## Visualizations
- Predictions table
- Class distribution
- Feature importance (top 20 TF-IDF features)

## Technologies
- Python 3.12
- scikit-learn
- pandas
- matplotlib
- seaborn
- Streamlit
- joblib

## Additional Files
- informe_final.md: Final report with results and reflections
- diagrama_solucion.png: Diagram explaining the solution design
- data/ejemplos/: Sample CSV files for testin

## License
- MIT License

## Installation

```bash
git clone https://github.com/Sporte25/BiometicaIA.git
cd BiometicaIA
pip install -r requirements.txt

