from preparation import load_data, separacio_train_test
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
from evaluation import get_best_model

# Definir la cuadrícula de parámetros para cada modelo
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

# Ruta al dataset preprocesado
DATA_PATH = "data/scaled_dataset.pkl"
TARGET_COLUMNS = ["estres"]  # Cambia por la columna objetivo

# Cargar el dataset
data = load_data(DATA_PATH)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = separacio_train_test(data, TARGET_COLUMNS)

# Resultados de importancia de características
results = {}

# Modelos
models = {
    "ridge": (Ridge(), GRID_PARAMS["ridge"]),
    "lasso": (Lasso(), GRID_PARAMS["lasso"]),
    "random_forest": (RandomForestRegressor(random_state=42), GRID_PARAMS["random_forest"]),
    "gradient_boosting": (GradientBoostingRegressor(random_state=42), GRID_PARAMS["gradient_boosting"]),
    "xgboost": (XGBRegressor(random_state=42), GRID_PARAMS["xgboost"])
}

# Entrenar y evaluar cada modelo
for model_name, (model, param_grid) in models.items():
    # Buscar el mejor modelo con GridSearchCV
    best_model = get_best_model(model_name, model, param_grid)
    best_model.fit(X_train, y_train)
    
    # Calcular la importancia de características
    if model_name in ["ridge", "lasso"]:
        coefficients = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': best_model.coef_
        }).sort_values(by='Coefficient', ascending=False)
        print(f"--- Importancia de características para {model_name} ---")
        print(coefficients)
        results[model_name] = coefficients

    elif model_name in ["random_forest", "gradient_boosting", "xgboost"]:
        importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print(f"--- Importancia de características para {model_name} ---")
        print(importances)
        results[model_name] = importances


# Guardar resultados
for method, importance_df in results.items():
    output_path = f"data/importance_{method}.xlsx"
    importance_df.to_excel(output_path, index=False)
    print(f"Guardado {method} en {output_path}")
