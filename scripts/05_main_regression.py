from preprocess import preprocess
from feature_engineering import features_importance,select_k_best_features,save_results
from evaluation_regression import evaluate_model,plot_error_by_class,plot_learning_curves,analyze_error_by_class, plot_metrics
from models_regression import RegressionModels
from preparation_regression import separacio_train_test
import pandas as pd
import os

# Configuració
TARGET_COLUMN = "estres"
DATA_PATH = preprocess('data/cleaned_dataset.pkl',TARGET_COLUMN)
FEATURES = ['ordenador', 'otrofactor', 'dayoftheweek','bienestar']
MODELS = ['svr', 'xgboost','polynomial_regression','random_forest','gradient_boosting']
RESULTS_DIR = "data/regression/final_results"


# Ejecución principal
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = separacio_train_test(DATA_PATH, TARGET_COLUMN)

    results = features_importance(MODELS, X_train, y_train,X_test,y_test)
    save_results(results)
    print("Feature importances guardadas")

    # Extraer las 10 características más importantes comunes
    common_features = select_k_best_features(X_train,y_train, 10)
    print("Las 10 características más importantes comunes entre los modelos son:")
    print(common_features)
    print("Evaluación completada.")

    # Crear directorio para resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)   

    # Agrupar las clases 9 y 10 en una sola clase
   
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)


    y_train_grouped = y_train.replace({10: 9})  # Cambiar clase 10 por 9 en el conjunto de entrenamiento
    y_test_grouped = y_test.replace({10: 9})    # Cambiar clase 10 por 9 en el conjunto de prueba
    y_train_grouped = y_train.replace({0: 1})  # Cambiar clase 0 por 1 en el conjunto de entrenamiento
    y_test_grouped = y_test.replace({0: 1})    # Cambiar clase 0 por 1 en el conjunto de prueba
    
    # Evaluar modelos seleccionados

    metrics_dict = {}

    for model_name in MODELS:
        # Evaluar el modelo con las clases agrupadas
        metrics = evaluate_model(model_name, X_train, X_test, y_train_grouped, y_test_grouped)
        metrics_dict[model_name] = metrics

        # Obtener predicciones del modelo seleccionado
        model_instance = RegressionModels(model_type=model_name).get_model()
        model_instance.fit(X_train, y_train_grouped)
        y_test_pred = model_instance.predict(X_test)

        # Analizar errores en clases específicas
        print(f"\nAnálisis de errores para el modelo {model_name}:")
        target_classes = [1, 2, 3, 4, 5, 6, 7,8,9]  # Clase 10 ya está incluida en la clase 9 y la 0 en la 1.
        error_results = analyze_error_by_class(y_test_grouped, y_test_pred, target_classes)
        plot_error_by_class(error_results, title=f"Errores para el modelo {model_name} con clases agrupadas")

    # Visualizar resultados globales
    plot_metrics(metrics_dict)
    best_model_instance = RegressionModels(model_type="xgboost").get_model()
    plot_learning_curves(best_model_instance, X_train, X_test, y_train_grouped, y_test_grouped)  # curva de aprendizaje del modelo finalmente escogido

    print("Evaluación y análisis completados. Revisa los resultados en el directorio de resultados.")


