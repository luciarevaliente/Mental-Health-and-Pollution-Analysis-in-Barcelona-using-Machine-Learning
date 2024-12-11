from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
from evaluation import get_best_model, GRID_PARAMS
from preparation import load_data, separacio_train_test
from preprocess import preprocess

# Cargar y preparar datos
DATA_PATH = "data/cleaned_dataset.pkl"
TARGET_COLUMNS = "estres"
data = preprocess(DATA_PATH, TARGET_COLUMNS)
X_train, X_test, y_train, y_test = separacio_train_test(data, TARGET_COLUMNS)

# Resultados de importancia
results = {}

# Modelos
models = {
    "ridge": (Ridge(), GRID_PARAMS["ridge"]),
    "lasso": (Lasso(), GRID_PARAMS["lasso"]),
    "random_forest": (RandomForestRegressor(random_state=42), GRID_PARAMS["random_forest"]),
    "gradient_boosting": (GradientBoostingRegressor(random_state=42), GRID_PARAMS["gradient_boosting"]),
    "xgboost": (XGBRegressor(random_state=42), GRID_PARAMS["xgboost"]),
    "svr": (SVR(), GRID_PARAMS["svr"]),
    "polynomial_regression": (Pipeline([
        ("polynomialfeatures", PolynomialFeatures()),
        ("linearregression", Lasso())
    ]), GRID_PARAMS["polynomial_regression"])
}

# Entrenar y evaluar modelos
for model_name, (model, param_grid) in models.items():
    best_model = get_best_model(model_name, model, param_grid)
    best_model.fit(X_train, y_train)

    # Importancia de características
    if model_name in ["ridge", "lasso"]:
        coefficients = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': abs(best_model.coef_)
        }).sort_values(by='Coefficient', ascending=False)
        results[model_name] = coefficients

    elif model_name in ["random_forest", "gradient_boosting", "xgboost"]:
        importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        results[model_name] = importances

    elif model_name == "svr":
        perm_importance = permutation_importance(best_model, X_train, y_train, random_state=42)
        importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': perm_importance.importances_mean
        }).sort_values(by='Importance', ascending=False)
        results[model_name] = importances

# Guardar resultados
for method, importance_df in results.items():
    output_path = f"data/regression/scaled_shuffle/importance_{method}_scaled.xlsx"
    importance_df.to_excel(output_path, index=False)
    print(f"Guardado {method} en {output_path}")
