
---

## 2. Informe final — Resultados y reflexiones

```markdown
# Informe Final — BiometicaIA

## Descripción del reto

Desarrollar una solución que clasifique artículos médicos en múltiples categorías, permitiendo carga de archivos CSV y visualización de componentes.

## Enfoque técnico

- Modelo: Random Forest con `MultiOutputClassifier`
- Vectorización: TF-IDF con 5000 features
- Evaluación: F1-score ponderado, matriz de confusión, clasificación multietiqueta
- Visualización: Streamlit con pestañas interactivas

## Resultados

- F1-score ponderado: 0.87 (ejemplo de prueba)
- Matriz de confusión visible en dashboard
- Distribución de clases y características importantes mostradas dinámicamente

## Visualización V0
- Se desarrolló una interfaz interactiva con Streamlit que permite cargar archivos CSV, realizar predicciones,
y visualizar la matriz de confusión, la distribución de clases y las características más relevantes del modelo.

**Nota** : se debe aclarar que en apartado de la entrega final se agregó un link de V0
(https://v0-text-classification-dashboard.vercel.app/), pero este es solo fue un dashboard
de ejemplo ya que no se alcanzó a integrar con la aplicación.

## Reflexiones

- El modelo es solo para textos en inglés
- Se enfrentaron errores de deserialización que se resolvieron modularizando la clase `ModeloMultietiqueta`
- Se aprendió a manejar clasificación multietiqueta con `MultiLabelBinarizer` y a estructurar proyectos reproducibles

## Evidencias

- Archivos de ejemplo con etiquetas verdaderas
- Visualizaciones generadas en `dashboard.py`
- Código modular en `src/` y `dashboard_app/`
