import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from models_regression import get_best_model
from models_regression import RegressionModels, GRID_PARAMS

# Configuración
FEATURES = ['ordenador', 'otrofactor', 'dayoftheweek', 'bienestar','covid_motor','alcohol','drogas']
TARGET_COLUMN = "estres"
RESULTS_DIR = "data/regression/final_results"

cleaned_dataset_path = 'data/cleaned_dataset.pkl'
codif_dataset_path = 'data/codif_dataset.pkl'

cleaned_dataset = pd.read_pickle(cleaned_dataset_path)
data = pd.read_pickle(codif_dataset_path)

# Verificar si la columna 'estres' está en el dataset limpio
if 'estres' in cleaned_dataset.columns:
    # Añadir la columna 'estres' al dataset codificado
    if cleaned_dataset.shape[0] == data.shape[0]:
        data['estres'] = cleaned_dataset['estres']
        print("Columna 'estres' añadida correctamente.")

# Crear directorio para resultados
os.makedirs(RESULTS_DIR, exist_ok=True)

# Función para evaluar el modelo

def evaluate_model(model_name, X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa un modelo, devolviendo métricas clave y guardando resultados.
    """
    print(f"Evaluando {model_name}...")
    model_instance = RegressionModels(model_type=model_name)
    param_grid = GRID_PARAMS.get(model_name, {})
    best_model = get_best_model(model_name, model_instance.get_model(), param_grid, X_train, y_train)

    # Entrenar el modelo
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

# Carga de datos

def load_and_prepare_data(data, features, target_column):
    """Carga y prepara los datos, seleccionando las características y el target."""
    
    X = data[features]
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

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

######################################################################################################################################
# MAIN

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # Cargar datos
    X_train, X_test, y_train, y_test = load_and_prepare_data(data, FEATURES, TARGET_COLUMN)

    # Evaluar modelos seleccionados
    model_types = [ "random_forest", "gradient_boosting", "xgboost"]
    metrics_dict = {}

    for model_name in model_types:
        metrics = evaluate_model(model_name, X_train, X_test, y_train, y_test)
        metrics_dict[model_name] = metrics

    # Visualizar resultados
    plot_metrics(metrics_dict)
    best_model_instance = RegressionModels(model_type="random_forest").get_model()
    plot_learning_curves(best_model_instance, X_train, X_test, y_train, y_test)

    print("Evaluación y análisis completados. Revisa los resultados en el directorio de resultados.")