# src/evaluacion.py
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    #plt.savefig('../resultados/matrices_confusion.png')
    import os
    os.makedirs("../resultados", exist_ok=True)
    plt.savefig("../resultados/matrices_confusion.png")
    plt.show()
    
    return {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'reporte': classification_report(y_true, y_pred, target_names=categorias, output_dict=True)
    }