import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from models_regression import get_best_model
from models_regression import RegressionModels, GRID_PARAMS
from preparation_regression import separacio_train_test
from preprocess import preprocess


# Función para evaluar el modelo

def evaluate_model(model_name, X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa un modelo, devolviendo métricas clave y guardando resultados.
    """
    print(f"Evaluando {model_name}...")
    
    model_instance = RegressionModels(model_type=model_name)
    param_grid = GRID_PARAMS.get(model_name, {})
    best_model = get_best_model(model_name, model_instance.get_model(), param_grid, X_train, y_train,X_test,y_test)
    

    # Entrenar el modelo
    if model_name == "xgboost":
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],  # Conjunto de validación
              # Número de iteraciones sin mejora antes de detenerse
            verbose=False)
        
    else:
        best_model.fit(X_train, y_train)

    # Predicciones
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calcular métricas
    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred)
    }

    # Guardar métricas
    metrics_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Métricas guardadas en {metrics_path}")

    # Guardar importancia de características
    if hasattr(best_model, "feature_importances_"):
        importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        importance_path = os.path.join(RESULTS_DIR, f"{model_name}_importance.csv")
        importances.to_csv(importance_path, index=False)
        print(f"Importancias guardadas en {importance_path}")

    return metrics

def analyze_error_by_class(y_test, y_test_pred, target_classes):
    """
    Analiza el error del modelo para clases específicas de la variable objetivo.

    Args:
        y_test (pd.Series): Valores reales de la variable objetivo.
        y_test_pred (np.array): Predicciones del modelo.
        target_classes (list): Clases específicas para analizar el error.

    Returns:
        dict: Métricas de error por clase.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    results = {}
    
    for target_class in target_classes:
        # Filtrar las muestras para la clase objetivo
        mask = y_test == target_class
        y_test_filtered = y_test[mask]
        y_test_pred_filtered = y_test_pred[mask]

        # Calcular métricas
        mse = mean_squared_error(y_test_filtered, y_test_pred_filtered) if len(y_test_filtered) > 0 else None
        mae = mean_absolute_error(y_test_filtered, y_test_pred_filtered) if len(y_test_filtered) > 0 else None

        # Guardar resultados
        results[target_class] = {"MSE": mse, "MAE": mae}

        # Mostrar errores
        print(f"Clase {target_class}:")
        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")

    
    return results


def plot_error_by_class(error_results, title="Errores por clase (MSE y MAE)"):
    """
    Genera un gráfico de barras mostrando el MSE y MAE por clase.

    Args:
        error_results (dict): Resultados de error por clase en formato {clase: {"MSE": valor, "MAE": valor}}.
        title (str): Título del gráfico.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extraer datos para el gráfico
    classes = list(error_results.keys())
    mse_values = [error_results[cls]["MSE"] if error_results[cls]["MSE"] is not None else 0 for cls in classes]
    mae_values = [error_results[cls]["MAE"] if error_results[cls]["MAE"] is not None else 0 for cls in classes]

    # Crear el gráfico
    x = np.arange(len(classes))  # Posiciones para las barras
    width = 0.4  # Ancho de las barras

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, mse_values, width, label='MSE', color='orange', alpha=0.7)
    plt.bar(x + width / 2, mae_values, width, label='MAE', color='blue', alpha=0.7)

    # Configurar etiquetas y leyenda
    plt.title(title)
    plt.xlabel("Clases")
    plt.ylabel("Error")
    plt.xticks(x, classes)
    plt.legend()

    # Mostrar el gráfico
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Visualizar métricas

def plot_metrics(metrics_dict):
    """
    Crea gráficos de barras comparando las métricas de diferentes modelos.
    """
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title("Comparación de métricas entre modelos")
    plt.ylabel("Valor")
    plt.xlabel("Modelos")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_metrics_comparison.png"))
    plt.show()


# Curvas de aprendizaje
def plot_learning_curves(model, X_train, X_test, y_train, y_test):
    """
    Genera curvas de aprendizaje mostrando el desempeño del modelo con diferentes tamaños de datos de entrenamiento.
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []

    for train_size in train_sizes:
        subset_X_train = X_train[:int(len(X_train) * train_size)]
        subset_y_train = y_train[:int(len(y_train) * train_size)]

        model.fit(subset_X_train, subset_y_train)
        train_pred = model.predict(subset_X_train)
        test_pred = model.predict(X_test)

        train_scores.append(mean_squared_error(subset_y_train, train_pred))
        test_scores.append(mean_squared_error(y_test, test_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, label="Train MSE")
    plt.plot(train_sizes, test_scores, label="Test MSE")
    plt.title("Curvas de aprendizaje")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"))
    plt.show()


    