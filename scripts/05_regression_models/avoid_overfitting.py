from preparation import load_data, separacio_train_test
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Modelos
models = {
    "ridge": (Ridge(), GRID_PARAMS["ridge"]),
    "lasso": (Lasso(), GRID_PARAMS["lasso"]),
    "random_forest": (RandomForestRegressor(random_state=42), GRID_PARAMS["random_forest"]),
    "gradient_boosting": (GradientBoostingRegressor(random_state=42), GRID_PARAMS["gradient_boosting"]),
    "xgboost": (XGBRegressor(random_state=42), GRID_PARAMS["xgboost"])
}

# Función para buscar el mejor modelo
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

# Función para visualizar predicciones
def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title(f"Predicciones vs Reales ({model_name})")
    plt.show()

# Evaluar modelos
results = []
for model_name, (model, param_grid) in models.items():
    # Dividir X_train en un conjunto de validación solo para xgboost
    if model_name == "xgboost":
        X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        print(f"Entrenando {model_name} con early stopping...")
        
        # Buscar el mejor modelo
        best_model = get_best_model(model_name, model, param_grid)
        
        # Entrenar con early stopping
        best_model.fit(
            X_train_main,
            y_train_main,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=True
        )
    else:
        # Buscar el mejor modelo para otros algoritmos
        best_model = get_best_model(model_name, model, param_grid)
        best_model.fit(X_train, y_train)
    
    # Validación cruzada
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    print(f"Validación cruzada (MSE) para {model_name}: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Predicción en el conjunto de prueba
    y_pred = best_model.predict(X_test)
    
    # Métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": model_name, "MSE": mse, "R2": r2})
    
    # Visualizar predicciones
    plot_predictions(y_test, y_pred, model_name)

# Mostrar resultados finales
results_df = pd.DataFrame(results)
print("\n--- Resultados finales ---")
print(results_df)

# Guardar resultados
results_df.to_excel("data/model_comparison_results.xlsx", index=False)
print("Resultados guardados en 'data/model_comparison_results.xlsx'")
