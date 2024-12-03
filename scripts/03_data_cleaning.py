"""
Script per natejar les dades de salut mental i contaminació de BCN
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 03/12/24
Descripció: Aquest script nateja les dades de salut mental i contaminació.
"""
# 03_data_cleaning.py
# IMPORTACIÓ
import pandas as pd
import numpy as np

# VARIABLES CONSTANTS
PICKLE_PATH = 'data/dataset.pkl'
CLEANED_PICKLE_PATH = 'data/cleaned_dataset.pkl'

# Cargar el DataFrame desde el archivo pickle guardado previamente
data = pd.read_pickle(PICKLE_PATH)
# print(data.isnull().sum())
print(data['performance'].mean())

# Imputar valores nulos
for column in data.columns:
    if data[column].isnull().any():  # Verifica si la columna tiene valores nulos
        if data[column].dtype == 'object' or data[column].dtype.name == 'category':  # Columnas categóricas
            mode_value = data[column].mode()[0]  # Calcular la moda
            data[column] = data[column].fillna(mode_value)  # Reasignar a la columna
        elif np.issubdtype(data[column].dtype, np.number):  # Columnas numéricas
            mean_value = data[column].mean()  # Calcular la media
            data[column] = data[column].fillna(mean_value)  # Reasignar a la columna

# Guardar el dataset imputado nuevamente en un archivo pickle
data.to_pickle(CLEANED_PICKLE_PATH)

# Confirmar que ya no hay valores nulos
# print(data.isnull().sum())
print(data['performance'].mean())
