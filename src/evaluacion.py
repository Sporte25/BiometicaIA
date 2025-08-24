# src/evaluacion.py
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluar_modelo_multietiqueta(y_true, y_pred, categorias):
    """
    Evaluación completa para problemas multi-etiqueta
    """
    print("=" * 50)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 50)
    
    # Métricas principales
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    
    # Precisión por categoría
    print("\nMétricas por categoría:")
    for i, categoria in enumerate(categorias):
        f1_cat = f1_score(y_true[:, i], y_pred[:, i])
        print(f"{categoria}: F1 = {f1_cat:.4f}")
    
    # Reporte de clasificación
    print("\n" + "=" * 50)
    print("REPORTE DE CLASIFICACIÓN")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=categorias))
    
    # Matrices de confusión por categoría
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, categoria in enumerate(categorias):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Matriz de Confusión - {categoria}')
        axes[i].set_xlabel('Predicho')
        axes[i].set_ylabel('Real')
    
    plt.tight_layout()
    
    # Crear carpeta resultados en la raíz
    resultados_dir = os.path.join(os.path.dirname(__file__), "../resultados")
    os.makedirs(resultados_dir, exist_ok=True)
    
    # Guardar archivo
    path_guardado = os.path.join(resultados_dir, "matrices_confusion.png")
    plt.savefig(path_guardado)
    plt.close(fig)

    print(f"\n✅ Imagen guardada en: {os.path.abspath(path_guardado)}")
    
    return {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'reporte': classification_report(y_true, y_pred, target_names=categorias, output_dict=True)
    }
