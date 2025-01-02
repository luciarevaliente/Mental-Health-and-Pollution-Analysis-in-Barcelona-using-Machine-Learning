from preprocess import preprocess
from feature_importance_regression import features_importance, select_k_best_features, save_results
from evaluation_regression import evaluate_model, plot_error_by_class, plot_learning_curves, analyze_error_by_class, plot_metrics
from models_regression import RegressionModels
from preparation_regression import separacio_train_test
import pandas as pd
import os

# Configuració
TARGET_COLUMN = "estres"  # Defineix el nom de la columna objectiu, que és "estres".
DATA_PATH = preprocess('data/cleaned_dataset.pkl', TARGET_COLUMN)  # Preprocessa el conjunt de dades i el carrega des d'un fitxer Pickle.
FEATURES = ['ordenador', 'otrofactor', 'dayoftheweek', 'bienestar']  # Llista de característiques seleccionades per a l'anàlisi.
MODELS = ['xgboost']# ['svr', 'xgboost', 'random_forest','polynomial_regression', , 'gradient_boosting']  # Tipus de models que s'utilitzaran.
RESULTS_DIR = "visualization/importance_&_metrics_regression"  # Ruta del directori on es desaran els resultats.

# Ejecució principal
if __name__ == "__main__":

    # Divideix les dades en conjunts de train i test
    X_train, X_test, y_train, y_test = separacio_train_test(DATA_PATH, TARGET_COLUMN)
    print('Buscant característiques importants')
    # Calcula la importància de les característiques utilitzant diversos models
    results = features_importance(MODELS, X_train, y_train, X_test, y_test)
    save_results(results,RESULTS_DIR)  # Desa els resultats en fitxers.
    print("Feature importances guardades")  # Informa que les importàncies s'han desat.

    # Selecciona les 10 característiques més importants
    common_features = select_k_best_features(X_train, y_train, 10)
    print("\nLas 10 características más importantes comunes entre los modelos son:")  # Missatge informatiu.
    print(common_features)  # Mostra les característiques seleccionades.
    
    # Crea el directori per desar resultats si no existeix
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Agrupa les classes 9 i 10 en una sola classe i la classe 0 en la classe 1
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)  # Converteix `y_train` en un pandas.Series si no ho és.
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)  # Converteix `y_test` en un pandas.Series si no ho és.

    y_train_grouped = y_train.replace({10: 9})  # Agrupa la classe 10 a la classe 9 en `y_train`.
    y_test_grouped = y_test.replace({10: 9})  # Agrupa la classe 10 a la classe 9 en `y_test`.
    y_train_grouped = y_train.replace({0: 1})  # Agrupa la classe 0 a la classe 1 en `y_train`.
    y_test_grouped = y_test.replace({0: 1})  # Agrupa la classe 0 a la classe 1 en `y_test`.

    # Inicialitza un diccionari per emmagatzemar les mètriques dels models
    metrics_dict = {}

    for model_name in MODELS:
        # Avalua el model utilitzant les dades agrupades
        metrics = evaluate_model(model_name, X_train, X_test, y_train_grouped, y_test_grouped, RESULTS_DIR)
        metrics_dict[model_name] = metrics  # Desa les mètriques al diccionari.

        # Entrena el model seleccionat
        model_instance = RegressionModels(model_type=model_name).get_model()
        model_instance.fit(X_train, y_train_grouped)

        # Genera prediccions amb el model entrenat
        y_test_pred = model_instance.predict(X_test)

        # Analitza els errors per classes específiques
        print(f"\nAnàlisi d'errors per al model {model_name}:")  # Missatge informatiu.
        target_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Defineix les classes a analitzar.
        error_results = analyze_error_by_class(y_test_grouped, y_test_pred, target_classes)
        plot_error_by_class(error_results, title=f"Errores para el modelo {model_name} con clases agrupadas")  # Mostra els errors en un gràfic.

    # Visualitza els resultats globals
    plot_metrics(metrics_dict,RESULTS_DIR)  # Mostra gràfics comparant les mètriques dels models.
    best_model_instance = RegressionModels(model_type="xgboost").get_model()  # Selecciona el model final (XGBoost).
    plot_learning_curves(best_model_instance, X_train, X_test, y_train_grouped, y_test_grouped,RESULTS_DIR)  # Genera les corbes d'aprenentatge del model final.

    print("Avaluació completada")  # Missatge final informatiu.
