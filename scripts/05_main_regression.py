from preprocess import preprocess
from evaluation_regression import features_importance,select_k_best_features,save_results, prepare_data

# Configuración

TARGET_COLUMN = "estres"
DATA_PATH = preprocess('data/cleaned_dataset.pkl',TARGET_COLUMN)


# Ejecución principal
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH, TARGET_COLUMN)
    model_types = [ "random_forest", "gradient_boosting", "xgboost"] # modelos para encontrar las características mas importantes
    results = features_importance(model_types, X_train, y_train)
    save_results(results)
    print("Feature imoportances guardadas")

    # Extraer las 10 características más importantes comunes
    common_features = select_k_best_features(X_train,y_train, 10)
    print("Las 10 características más importantes comunes entre los modelos son:")
    print(common_features)
    print("Evaluación completada.")
