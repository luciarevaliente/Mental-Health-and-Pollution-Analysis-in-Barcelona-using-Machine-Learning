from preparation import load_data, separacio_train_test
from models import RegressionModels
from evaluation import evaluate_model

# VARIABLES CONSTANTS
DATA_PATH = "data/salud_mental.csv"
TARGET_COLUMNS = ["estrés", "energía", "sueño"]
MODEL_TYPES = ["linear", "ridge", "lasso", "random_forest", "xgboost"]

def main():
    # Carrega les dades
    data = load_data(DATA_PATH)

    # Preprocesa
    X_train, X_test, y_train, y_test = separacio_train_test(data, TARGET_COLUMNS)

    # Entrena i evalua cada model    
    results = {}

    for model_type in MODEL_TYPES:
        print(f"Evaluando modelo: {model_type}")
        model = RegressionModels(model_type=model_type)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        results[model_type] = evaluate_model(y_test, predictions)
    
    # Mostrem resultats
    for model, metrics in results.items():
        print(f"Modelo: {model}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
