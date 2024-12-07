from preparation import *
from models import *
from evaluation import *
from sklearn.model_selection import GridSearchCV

# VARIABLES CONSTANTS
DATA_PATH = "data/cleaned_dataset.pkl"
TARGET_COLUMNS = ["estres"]
MODEL_TYPES = ["linear", "ridge", "lasso", "random_forest", "xgboost"]
GRID_PARAMS = {
    "linear": {},  # No requiere hiperparámetros ajustables
    "ridge": {"alpha": [0.1, 1.0, 10.0]},
    "lasso": {"alpha": [0.01, 0.1, 1.0]},
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }
}

def main():
    # Carga los datos
    data = load_data(DATA_PATH)

    # Preprocesa
    X_train, X_test, y_train, y_test = separacio_train_test(data, TARGET_COLUMNS)

    # Resultados
    results = {}

    for model_type in MODEL_TYPES:
        print(f"Buscando mejores parámetros para el modelo: {model_type}")

        # Obtén el modelo base y los parámetros de la cuadrícula
        params = GRID_PARAMS.get(model_type, {})
        base_model = RegressionModels(model_type=model_type).get_model()

        # Configura GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=params,
            scoring="neg_mean_squared_error",  # Métrica de evaluación
            cv=5,  # Validación cruzada
            verbose=1,  # Nivel de detalle del proceso
            n_jobs=-1  # Usa todos los núcleos disponibles
        )

        # Ajusta el modelo con búsqueda de hiperparámetros
        grid_search.fit(X_train, y_train)  # Asegura que y_train sea unidimensional

        # Obtén el mejor modelo y evalúa en el conjunto de prueba
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        results[model_type] = evaluate_model(y_test, predictions)

        # Imprime el mejor conjunto de hiperparámetros
        print(f"Mejores parámetros para {model_type}: {grid_search.best_params_}")

    # Mostrar resultados finales
    for model, metrics in results.items():
        print(f"Modelo: {model}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__=="__main__":
    main()