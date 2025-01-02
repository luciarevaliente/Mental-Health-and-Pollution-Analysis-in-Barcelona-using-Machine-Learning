from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(file_path):
    """
    Carga los datos desde un archivo Pickle.
    
    Args:
        file_path (string): String amb el path del pickle a carregar.

    Returns:
        DataFrame: DataFrame amb el dataset per entrenar el model.
    """
    return pd.read_pickle(file_path)

def separacio_train_test(data, target_columns):
    """
    Args:
        data (DataFrame): Datos preprocesados.
        target_columns (list): Columnas objetivo para predecir.

    Returns:
        X_train, X_test, y_train, y_test: DataFrames para entrenamiento y prueba.
    """
    X = data.drop(columns=target_columns)
    y = data[target_columns]

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir y_train e y_test a arrays unidimensionales
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    return X_train, X_test, y_train, y_test

