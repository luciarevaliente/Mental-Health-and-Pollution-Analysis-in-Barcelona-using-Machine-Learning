from evaluation_regression import get_best_model, GRID_PARAMS
from preparation_regression import load_data, separacio_train_test
from preprocess import preprocess
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
import os


# Configuración del dataset y las columnas objetivo
DATA_PATH = "data/cleaned_dataset.pkl"
TARGET_COLUMNS = "estres"

# Preparación de datos
def prepare_data(data_path, target_column):
    data = preprocess(data_path, target_column)
    return separacio_train_test(data, target_column)

X_train, X_test, y_train, y_test = prepare_data(DATA_PATH, TARGET_COLUMNS)

# Modelos y configuración
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

# Evaluar modelos y calcular importancia de características
def evaluate_models(models, X_train, y_train):
    results = {}
    count=0
    for model_name, (model, param_grid) in models.items():
        count+=1
        print('buscando',count)
        best_model = get_best_model(model_name, model, param_grid)
        best_model.fit(X_train, y_train)

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
            print('aquiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
            perm_importance = permutation_importance(best_model, X_train, y_train, random_state=42)
            importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': perm_importance.importances_mean
            }).sort_values(by='Importance', ascending=False)
            results[model_name] = importances
    return results

# Guardar resultados en Excel
def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for method, importance_df in results.items():
        output_path = os.path.join(output_dir, f"importance_{method}_scaled.xlsx")
        importance_df.to_excel(output_path, index=False)
        print(f"Guardado {method} en {output_path}")

# Extraer las características comunes más importantes
def extract_common_features(results, top_n=10):
    top_features_sets = []
    for importance_df in results.values():
        top_features = set(importance_df.head(top_n)['Feature'])
        top_features_sets.append(top_features)
    common_features = set.intersection(*top_features_sets)
    return common_features

# Ejecución principal
if __name__ == "__main__":
    results = evaluate_models(models, X_train, y_train)
    # save_results(results, "data/regression/scaled_shuffle/")
    # Imprimir las 5 características más importantes comunes
    common_features = extract_common_features(results, top_n=5)
    print("Las 5 características más importantes comunes son:")
    print(common_features)
