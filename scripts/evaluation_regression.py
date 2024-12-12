from preparation_regression import load_data, separacio_train_test
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models_regression import RegressionModels
import matplotlib.pyplot as plt
import numpy as np

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
    },
    "svr": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "epsilon": [0.01, 0.1, 1.0]
    },
    "polynomial_regression": {
        "polynomialfeatures__degree": [2]
    }
}

# Ruta al dataset preprocesado
DATA_PATH = "data/scaled_dataset.pkl"
TARGET_COLUMNS = ["estres"]  # Cambia por la columna objetivo

# Cargar el dataset
data = load_data(DATA_PATH)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = separacio_train_test(data, TARGET_COLUMNS)

# Resultados de importancia de características
results = {}

# Evaluar modelos
def evaluate_model(y_test, predictions, model_name):
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"\n--- Error del modelo: {model_name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
  
    
    return {"Model": model_name, "RMSE": rmse, "MAE": mae}

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


# Función para obtener el mejor modelo con GridSearchCV

def get_best_model(model_name, base_model, param_grid):
    print(f"\n--- Buscando mejores parámetros para {model_name} ---")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        scoring="neg_mean_squared_error",
        n_iter=10,  # Limitar a 10 combinaciones
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    print(f"Mejores parámetros para {model_name}: {random_search.best_params_}")
    return random_search.best_estimator_


# def cross_validate_model(model, X_train, y_train):
#     """
#     Realiza validación cruzada y devuelve el puntaje promedio.

#     Args:
#         model: Modelo a validar.
#         X_train (DataFrame): Datos de entrenamiento.
#         y_train (Series): Valores objetivo de entrenamiento.

#     Returns:
#         float: Puntaje promedio de validación cruzada.
#     """
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
#     mean_score = -cv_scores.mean()
#     print(f"Validación cruzada (MSE promedio): {mean_score:.4f} (+/- {cv_scores.std():.4f})")
#     return mean_score

# # Modelos a evaluar
# model_types = ["ridge", "lasso", "random_forest", "gradient_boosting", "xgboost","svr", "polynomial_regression"]

# performance = []
# predictions_dict = {}

# # for model_type in model_types:
# #     # Obtener modelo base y parámetros
# #     base_model = RegressionModels(model_type=model_type).get_model()
# #     param_grid = GRID_PARAMS.get(model_type, {})
    
# #     # Buscar el mejor modelo con GridSearchCV
# #     best_model = get_best_model(model_type, base_model, param_grid)
# #     best_model.fit(X_train, y_train)
    
# #     # Predicciones
# #     predictions = best_model.predict(X_test)
    
# #     # Evaluar y guardar resultados
# #     performance.append(evaluate_model(y_test, predictions, model_type))
# #     predictions_dict[model_type] = predictions

# # # Comparación de métricas
# # performance_df = pd.DataFrame(performance)
# # print("\n--- Comparación de modelos ---")
# # print(performance_df)

# # # Visualizar métricas
# performance_df.plot(x="Model", y=["RMSE", "MAE"], kind="bar", figsize=(10, 6), rot=0, title="Comparación de Modelos")

# # Visualizar predicciones
# plot_real_vs_pred_multiple(y_test, predictions_dict)
