from sklearn.feature_selection import SelectKBest, f_regression
from models_regression import RegressionModels, GRID_PARAMS, get_best_model
import pandas as pd
import os

# Evaluar modelos y calcular importancia de características
def features_importance(model_types, X_train, y_train,X_test,y_test):
     results = {}
     for model_name in model_types:
         print(f"Evaluando {model_name}...")
         model_instance = RegressionModels(model_type=model_name)
         param_grid = GRID_PARAMS.get(model_name, {})
         best_model = get_best_model(model_name, model_instance.get_model(), param_grid,X_train, y_train,X_test, y_test)
         best_model.fit(X_train, y_train)
         # Obtener importancia de características
         if hasattr(best_model, "coef_"):  # Modelos lineales
             coefficients = pd.DataFrame({
                 'Feature': X_train.columns,
                 'Coefficient': abs(best_model.coef_)
             }).sort_values(by='Coefficient', ascending=False)
             results[model_name] = coefficients

         elif hasattr(best_model, "feature_importances_"):  # Modelos de árboles
             importances = pd.DataFrame({
                 'Feature': X_train.columns,
                 'Importance': best_model.feature_importances_
             }).sort_values(by='Importance', ascending=False)
             results[model_name] = importances

         else:  # Modelos sin coeficientes directos
             from sklearn.inspection import permutation_importance
             perm_importance = permutation_importance(best_model, X_train, y_train, random_state=42)
             importances = pd.DataFrame({
                 'Feature': X_train.columns,
                 'Importance': perm_importance.importances_mean
             }).sort_values(by='Importance', ascending=False)
             results[model_name] = importances
     return results

def select_k_best_features(X, y, k=10):
    """Selecciona las k mejores características."""
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features

# Guardar resultados
def save_results(results, output_dir="data/regression/results"):
    os.makedirs(output_dir, exist_ok=True)
    for model_name, importance_df in results.items():
        importance_df.to_excel(os.path.join(output_dir, f"{model_name}_importance.xlsx"), index=False)
        print(f"Resultados guardados para {model_name} en {output_dir}")