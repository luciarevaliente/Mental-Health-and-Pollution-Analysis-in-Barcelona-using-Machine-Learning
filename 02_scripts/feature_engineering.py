from sklearn.feature_selection import SelectKBest, f_regression  # Llibreria per seleccionar les millors característiques basades en estadístiques.
from models_regression import RegressionModels, GRID_PARAMS, get_best_model  # Models de regressió, paràmetres de cerca i funció per obtenir el millor model.
import pandas as pd  # Llibreria per manipular dades tabulars.
import os  # Llibreria per gestionar el sistema de fitxers.

# Funció per avaluar models i calcular la importància de les característiques
def features_importance(model_types, X_train, y_train, X_test, y_test):
    """
    Avalua models i calcula la importància de les característiques segons diferents tipus de models.

    Args:
        model_types (list): Llista dels tipus de models a avaluar.
        X_train (DataFrame): Dades d'entrenament.
        y_train (Series): Variable objectiu per a l'entrenament.
        X_test (DataFrame): Dades de test.
        y_test (Series): Variable objectiu per al test.

    Returns:
        dict: Diccionari amb la importància de les característiques per a cada model.
    """
    results = {}  # Inicialitza un diccionari per emmagatzemar els resultats.

    for model_name in model_types:  # Itera per cada tipus de model especificat.
        print(f"Evaluando {model_name}...")  # Mostra quin model s'està avaluant.
        model_instance = RegressionModels(model_type=model_name)  # Crea una instància del model especificat.
        param_grid = GRID_PARAMS.get(model_name, {})  # Obté els paràmetres de cerca per al model.
        best_model = get_best_model(model_name, model_instance.get_model(), param_grid, X_train, y_train, X_test, y_test)  # Busca el millor model amb hiperparàmetres ajustats.
        best_model.fit(X_train, y_train)  # Entrena el model amb les dades d'entrenament.

        # Obté la importància de les característiques
        if hasattr(best_model, "coef_"):  # Si el model té coeficients (models lineals).
            coefficients = pd.DataFrame({
                'Feature': X_train.columns,  # Nom de les característiques.
                'Coefficient': abs(best_model.coef_)  # Valor absolut dels coeficients.
            }).sort_values(by='Coefficient', ascending=False)  # Ordena les característiques per importància.
            results[model_name] = coefficients  # Desa els resultats al diccionari.

        elif hasattr(best_model, "feature_importances_"):  # Si el model té atributs d'importància de característiques (models d'arbres).
            importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': best_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)  # Ordena les característiques per importància.
            results[model_name] = importances

        else:  # Si el model no té atributs directes per la importància de les característiques.
            from sklearn.inspection import permutation_importance  # Importa inspecció per permutacions.
            perm_importance = permutation_importance(best_model, X_train, y_train, random_state=42)  # Calcula la importància per permutació.
            importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': perm_importance.importances_mean  # Importància mitjana per permutació.
            }).sort_values(by='Importance', ascending=False)
            results[model_name] = importances  # Desa els resultats al diccionari.

    return results  # Retorna el diccionari amb els resultats.

# Funció per seleccionar les millors característiques
def select_k_best_features(X, y, k=10):
    """
    Selecciona les k millors característiques utilitzant estadístiques ANOVA F-test.

    Args:
        X (DataFrame): Conjunt de dades amb les característiques.
        y (Series): Variable objectiu.
        k (int): Nombre de característiques a seleccionar.

    Returns:
        tuple: Dades seleccionades (X_selected) i noms de les característiques seleccionades.
    """
    selector = SelectKBest(score_func=f_regression, k=k)  # Inicialitza el selector de característiques.
    X_selected = selector.fit_transform(X, y)  # Ajusta el selector i transforma les dades.
    selected_features = X.columns[selector.get_support()]  # Obté els noms de les característiques seleccionades.
    return X_selected, selected_features  # Retorna les dades seleccionades i els noms de les característiques.

# Funció per desar els resultats
def save_results(results, output_dir):
    """
    Desa els resultats d'importància de característiques en fitxers Excel.

    Args:
        results (dict): Diccionari amb els resultats d'importància de característiques per model.
        output_dir (str): Directori on es desaran els resultats.
    """
    os.makedirs(output_dir, exist_ok=True)  # Crea el directori si no existeix.

    for model_name, importance_df in results.items():  # Itera pels resultats de cada model.
        importance_df.to_excel(os.path.join(output_dir, f"{model_name}_importance.xlsx"), index=False)  # Desa els resultats en un fitxer Excel.
        print(f"Resultados guardados para {model_name} en {output_dir}")  # Informa que els resultats s'han desat correctament.
