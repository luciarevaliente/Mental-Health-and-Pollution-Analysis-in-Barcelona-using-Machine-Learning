import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Mòduls per calcular mètriques d'error.
import matplotlib.pyplot as plt  # Llibreria per crear gràfics.
from models_regression import get_best_model  # Funció per obtenir el millor model amb hiperparàmetres ajustats.
from models_regression import RegressionModels, GRID_PARAMS  # Classe de models de regressió i paràmetres de cerca.
from preparation_regression import separacio_train_test  # Funció per dividir les dades en train i test.
from preprocess import preprocess  # Funció per preprocessar les dades.
from imblearn.over_sampling import RandomOverSampler, SMOTE  # Llibreria per fer resampling de dades.
from imblearn.under_sampling import RandomUnderSampler

# Funció per avaluar el model
def evaluate_model(model_name, X_train, X_test, y_train, y_test, RESULTS_DIR):
    """
    Entrena i avalua un model, retornant mètriques clau i desant resultats.
    """
    print(f"Evaluando {model_name}...")  # Imprimeix el nom del model que s'està avaluant.

    model_instance = RegressionModels(model_type=model_name)  # Inicialitza una instància del model especificat.
    param_grid = GRID_PARAMS.get(model_name, {})  # Obté els paràmetres de cerca associats al model.
    best_model = get_best_model(model_name, model_instance.get_model(), param_grid, X_train, y_train, X_test, y_test)  # Busca el millor model amb els paràmetres.

    # Entrena el model
    if model_name == "xgboost":  # Si el model és XGBoost, entrena utilitzant conjunts de validació.
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],  # Conjunt de validació.
            verbose=False
        )
    else:  # Per a altres models, entrena normalment.
        best_model.fit(X_train, y_train)

    # Prediccions
    y_train_pred = best_model.predict(X_train)  # Prediccions per al conjunt d'entrenament.
    y_test_pred = best_model.predict(X_test)  # Prediccions per al conjunt de prova.

    # Calcula mètriques
    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),  # Error quadràtic mitjà per a train.
        "test_mse": mean_squared_error(y_test, y_test_pred),  # Error quadràtic mitjà per a test.
        "test_mae": mean_absolute_error(y_test, y_test_pred)  # Error absolut mitjà per a test.
    }

    # Desa les mètriques en un fitxer CSV
    metrics_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Métricas guardadas en {metrics_path}")  # Informa que s'han desat les mètriques.

    # Desa la importància de les característiques
    if hasattr(best_model, "feature_importances_"):  # Comprova si el model té atributs d'importància.
        importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        importance_path = os.path.join(RESULTS_DIR, f"{model_name}_importance.csv")
        importances.to_csv(importance_path, index=False)  # Desa les importàncies en un fitxer CSV.
        print(f"Importancias guardadas en {importance_path}")

    return metrics  # Retorna les mètriques calculades.


def resample_data(X, y, method="oversample"): # NO ENS VA MILLORAR L'ERROR! (methods)
    """
    Aplica resampling per balancejar les dades.

    Args:
        X (DataFrame o ndarray): Conjunt de dades d'entrenament amb les característiques.
        y (Series o ndarray): Conjunt de dades d'entrenament amb la variable objectiu.
        method (str): Mètode de resampling a aplicar. Pot ser "oversample", "undersample" o "smote".

    Returns:
        X_resampled, y_resampled: Conjunts de dades balancejats.
    """
    if method == "oversample":
        sampler = RandomOverSampler(random_state=42)  # Inicialitza un RandomOverSampler per generar més mostres de classes minoritàries.
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=42)  # Inicialitza un RandomUnderSampler per reduir mostres de classes majoritàries.
    elif method == "smote":
        sampler = SMOTE(random_state=42)  # Inicialitza SMOTE per generar mostres sintètiques per les classes minoritàries.
    else:
        raise ValueError("El mètode ha de ser 'oversample', 'undersample' o 'smote'.")
        
    X_resampled, y_resampled = sampler.fit_resample(X, y)  # Aplica el resampling.
    print(f"Resampling completat amb el mètode: {method}.")
    return X_resampled, y_resampled

def analyze_error_by_class(y_test, y_test_pred, target_classes):
    """
    Analitza l'error del model per a classes específiques de la variable objectiu.
    """
    results = {}  # Diccionari per emmagatzemar els resultats per classe.

    for target_class in target_classes:  # Itera per cada classe objectiu.
        mask = y_test == target_class  # Filtra les dades per la classe actual.
        y_test_filtered = y_test[mask]  # Mostres reals de la classe.
        y_test_pred_filtered = y_test_pred[mask]  # Prediccions per la classe.

        # Calcula mètriques d'error per la classe
        mse = mean_squared_error(y_test_filtered, y_test_pred_filtered) if len(y_test_filtered) > 0 else None
        mae = mean_absolute_error(y_test_filtered, y_test_pred_filtered) if len(y_test_filtered) > 0 else None

        # Desa els resultats
        results[target_class] = {"MSE": mse, "MAE": mae}

        # Mostra errors per classe
        print(f"Clase {target_class}:")
        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")

    return results  # Retorna el diccionari amb les mètriques per classe.

def plot_error_by_class(error_results, title="Errores por clase (MSE y MAE)"):
    """
    Genera un gràfic de barres mostrant el MSE i MAE per classe.
    """
    # Extrau dades per al gràfic
    classes = list(error_results.keys())
    mse_values = [error_results[cls]["MSE"] if error_results[cls]["MSE"] is not None else 0 for cls in classes]
    mae_values = [error_results[cls]["MAE"] if error_results[cls]["MAE"] is not None else 0 for cls in classes]

    # Crear el gràfic
    x = np.arange(len(classes))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, mse_values, width, label='MSE', color='orange', alpha=0.7)
    plt.bar(x + width / 2, mae_values, width, label='MAE', color='blue', alpha=0.7)

    plt.title(title)
    plt.xlabel("Clases")
    plt.ylabel("Error")
    plt.xticks(x, classes)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_metrics(metrics_dict, RESULTS_DIR):
    """
    Crea gràfics de barres comparant les mètriques dels models.
    """
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title("Comparación de métricas entre modelos")
    plt.ylabel("Valor")
    plt.xlabel("Modelos")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_metrics_comparison.png"))
    plt.show()

