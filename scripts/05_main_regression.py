from evaluation_regression import get_best_model
from preparation_regression import load_data, separacio_train_test
from models_regression import RegressionModels, GRID_PARAMS, get_best_model
import pandas as pd
from preprocess import preprocess
import os
from sklearn.feature_selection import SelectKBest, f_regression
from evaluation_regression import features_importance,select_k_best_features,save_results, prepare_data

# Configuración

TARGET_COLUMN = "estres"
DATA_PATH = preprocess('data/cleaned_dataset.pkl',TARGET_COLUMN)


# Ejecución principal
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH, TARGET_COLUMN)
    model_types = [ "random_forest", "gradient_boosting", "xgboost"]
    results = features_importance(model_types, X_train, y_train)
    save_results(results)
    # print("Evaluación completada.")

    # Extraer las 10 características más importantes comunes
    common_features = select_k_best_features(X_train,y_train, 10)
    print("Las 10 características más importantes comunes entre los modelos son:")
    print(common_features)
    print("Evaluación completada.")
