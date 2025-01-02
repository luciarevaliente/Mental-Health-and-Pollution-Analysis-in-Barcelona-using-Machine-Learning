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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carga el dataset (sustituye 'your_dataset.csv' con el nombre del archivo)
data = pd.read_csv('data\CitieSHealth_BCN_DATA_PanelStudy_20220414.csv')

# Especifica la columna target
target_column = 'estres'  # Sustituye con el nombre de tu columna target

# Verifica los primeros valores
print(data[target_column].head())

# Visualización con Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(data[target_column], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribució de la variable estres', fontsize=16)
plt.xlabel('Valor del target', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Visualización con Seaborn
plt.figure(figsize=(8, 6))
sns.histplot(data[target_column], bins=30, kde=True, color='blue', alpha=0.6)
plt.title('Distribución de la variable objetivo con KDE', fontsize=16)
plt.xlabel('Valor del target', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
