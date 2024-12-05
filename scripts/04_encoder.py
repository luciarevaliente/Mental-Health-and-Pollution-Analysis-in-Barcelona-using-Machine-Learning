from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import datawig

PICKLE_PATH = 'data/dataset.pkl'

# Cargar el dataset
data = pd.read_pickle(PICKLE_PATH)

# Copia del DataFrame
data_cleaned = data.copy()

# Iterar por columnas y aplicar imputación con DataWig
for column in data_cleaned.columns:
    if data_cleaned[column].isnull().sum() > 0:  # Si hay valores nulos en la columna
        # Configurar las columnas de entrada (todas las demás columnas) y la columna de salida (actual)
        input_columns = [col for col in data_cleaned.columns if col != column]
        output_column = column

        # Crear el imputador de DataWig
        imputer = datawig.SimpleImputer(
            input_columns=input_columns,  # Columnas de entrada
            output_column=output_column   # Columna objetivo a rellenar
        )

        # Entrenar el modelo
        imputer.fit()

        # Rellenar los valores nulos de la columna actual
        data_cleaned = imputer.predict(data_cleaned)

# Resultado
print("Dataset con valores nulos imputados:")
print(data_cleaned.head())

# Aplicar One-Hot Encoding a las columnas categóricas
# data_ohe = pd.get_dummies(data_cleaned, drop_first=True)
# print("\nDataset después de One-Hot Encoding:")
# print(data_ohe.head())


