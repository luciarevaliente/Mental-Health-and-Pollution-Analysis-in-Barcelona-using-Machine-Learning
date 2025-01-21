import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Mòduls per calcular mètriques d'error.
import matplotlib.pyplot as plt  # Llibreria per crear gràfics.
from models_regression import get_best_model  # Funció per obtenir el millor model amb hiperparàmetres ajustats.
from models_regression import RegressionModels, GRID_PARAMS  # Classe de models de regressió i paràmetres de cerca.
from preprocess import preprocess  # Funció per preprocessar les dades.
from imblearn.over_sampling import RandomOverSampler, SMOTE  # Llibreria per fer resampling de dades.
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Funció per avaluar un model.
def evaluate_model(model_name, X_train, X_test, y_train, y_test, RESULTS_DIR):
    print(f"Avaluant {model_name}...")  # Imprimeix el nom del model que s'està avaluant.
    
    # Inicialitza una instància del model especificat.
    model_instance = RegressionModels(model_type=model_name)
    
    # Obté els paràmetres de cerca associats al model.
    param_grid = GRID_PARAMS.get(model_name, {})
    
    # Busca el millor model utilitzant els paràmetres de cerca.
    best_model = get_best_model(model_name, model_instance.get_model(), param_grid, X_train, y_train, X_test, y_test)
    
    # Entrena el model.
    if model_name == "xgboost":  # Si el model és XGBoost, entrena utilitzant conjunts de validació.
        best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:  # Per a altres models, entrena normalment.
        best_model.fit(X_train, y_train)
    
    # Prediccions.
    y_train_pred = best_model.predict(X_train)  # Prediccions per al conjunt d'entrenament.
    y_test_pred = best_model.predict(X_test)  # Prediccions per al conjunt de prova.
    
    # Calcula mètriques.
    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred)
    }
    
    # Desa les mètriques en un fitxer CSV.
    metrics_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Mètriques desades a {metrics_path}")
    
    # Desa la importància de les característiques (si el model les proporciona).
    if hasattr(best_model, "feature_importances_"):
        importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        importance_path = os.path.join(RESULTS_DIR, f"{model_name}_importance.csv")
        importances.to_csv(importance_path, index=False)
        print(f"Importàncies desades a {importance_path}")
    
    return metrics  # Retorna les mètriques calculades.

# Funció per fer resampling de les dades.
def resample_data(X, y, method="oversample"):
    if method == "oversample":
        sampler = RandomOverSampler(random_state=42)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=42)
    elif method == "smote":
        sampler = SMOTE(random_state=42)
    else:
        raise ValueError("El mètode ha de ser 'oversample', 'undersample' o 'smote'.")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    print(f"Resampling completat amb el mètode: {method}.")
    return X_resampled, y_resampled

# Funció per analitzar l'error per classe.
def analyze_error_by_class(y_test, y_test_pred, target_classes):
    results = {}
    for target_class in target_classes:
        mask = y_test == target_class
        y_test_filtered = y_test[mask]
        y_test_pred_filtered = y_test_pred[mask]
        mse = mean_squared_error(y_test_filtered, y_test_pred_filtered) if len(y_test_filtered) > 0 else None
        mae = mean_absolute_error(y_test_filtered, y_test_pred_filtered) if len(y_test_filtered) > 0 else None
        results[target_class] = {"MSE": mse, "MAE": mae}
        print(f"Classe {target_class}:")
        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")
    return results

# Funció per generar un gràfic d'errors per classe.
def plot_error_by_class(error_results, title="Errors per classe (MSE i MAE)"):
    classes = list(error_results.keys())
    mse_values = [error_results[cls]["MSE"] if error_results[cls]["MSE"] is not None else 0 for cls in classes]
    mae_values = [error_results[cls]["MAE"] if error_results[cls]["MAE"] is not None else 0 for cls in classes]
    x = np.arange(len(classes))
    width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, mse_values, width, label='MSE', alpha=0.7)
    plt.bar(x + width / 2, mae_values, width, label='MAE', alpha=0.7)
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Error")
    plt.xticks(x, classes)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Funció per crear gràfics comparant les mètriques dels models.
def plot_metrics(metrics_dict, RESULTS_DIR):
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title("Comparació de mètriques entre models")
    plt.ylabel("Valor")
    plt.xlabel("Models")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_metrics_comparison.png"))
    plt.show()

# Funció per carregar dades des d'un fitxer Pickle.
def load_data(file_path):
    return pd.read_pickle(file_path)

# Funció per dividir les dades en entrenament i prova.
def separacio_train_test(data, target_columns):
    X = data.drop(columns=target_columns)
    y = data[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    return X_train, X_test, y_train, y_test
