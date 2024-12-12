from scripts.preparation_regression import load_data, separacio_train_test
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.models_regression import RegressionModels
import matplotlib.pyplot as plt
import numpy as np

# Ruta al dataset preprocesado
DATA_PATH = "data/complete_scaled_dataset.pkl"
TARGET_COLUMNS = ["estres"]  # Cambia por la columna objetivo

# Cargar el dataset
data = load_data(DATA_PATH)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = separacio_train_test(data, TARGET_COLUMNS)

# Resultados de importancia de características
results = {}
GRID_PARAMS = {
    "ridge": {"alpha": [0.1, 1.0, 10.0]},
    "lasso": {"alpha": [0.01, 0.1, 1.0]},
    "random_forest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6, 10]
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }
}

# Evaluar modelos
def evaluate_model(y_test, predictions, model_name):
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
#     RMSE (Root Mean Squared Error): Penaliza grandes errores. Más bajo es mejor.
# MAE (Mean Absolute Error): Promedio de errores absolutos. Más bajo es mejor.
# R² (Coeficiente de determinación): Mide qué tan bien el modelo explica la varianza en los datos. Más cercano a 1 es mejor.
    print(f"\n--- Desempeño del modelo: {model_name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

# Visualizaciones
def plot_real_vs_pred_multiple(y_test, predictions_dict):
    plt.figure(figsize=(10, 6))
    for model_name, predictions in predictions_dict.items():
        plt.scatter(y_test, predictions, alpha=0.6, label=model_name)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', linewidth=2, label='Ajuste perfecto')
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Comparación de Predicciones por Modelo")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals_multiple(y_test, predictions_dict):
    plt.figure(figsize=(10, 6))
    for model_name, predictions in predictions_dict.items():
        residuals = y_test - predictions
        plt.scatter(predictions, residuals, alpha=0.6, label=model_name)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Sin Error')
    plt.xlabel("Predicciones")
    plt.ylabel("Residuales")
    plt.title("Residuales por Modelo. Dataset complete_scaled")
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para obtener el mejor modelo con GridSearchCV
def get_best_model(model_name, base_model, param_grid):
    print(f"\n--- Buscando mejores parámetros para {model_name} ---")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Mejores parámetros para {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Modelos a evaluar
model_types = ["ridge", "lasso", "random_forest", "gradient_boosting", "xgboost"]

performance = []
predictions_dict = {}

for model_type in model_types:
    # Obtener modelo base y parámetros
    base_model = RegressionModels(model_type=model_type).get_model()
    param_grid = GRID_PARAMS.get(model_type, {})
    
    # Buscar el mejor modelo con GridSearchCV
    best_model = get_best_model(model_type, base_model, param_grid)
    best_model.fit(X_train, y_train)
    
    # Predicciones sobre el conjunto de entrenamiento
    predictions_train = best_model.predict(X_train)
    
    # Evaluar y guardar resultados (sobre entrenamiento)
    performance.append(evaluate_model(y_train, predictions_train, f"{model_type} (Train)"))
    predictions_dict[f"{model_type} (Train)"] = predictions_train

# Comparación de métricas sobre el conjunto de entrenamiento
performance_df_train = pd.DataFrame(performance)
print("\n--- Comparación de modelos (Train) ---")
print(performance_df_train)

# Visualizar métricas sobre entrenamiento
performance_df_train.plot(x="Model", y=["RMSE", "MAE", "R2"], kind="bar", figsize=(10, 6), rot=0, title="Comparación de Modelos (Train). Dataset complete_scaled")

# Visualizar predicciones sobre el conjunto de entrenamiento
plot_real_vs_pred_multiple(y_train, predictions_dict)

# Visualizar residuales sobre el conjunto de entrenamiento
plot_residuals_multiple(y_train, predictions_dict)
