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

# CARREGAR EL DATASET #############################################################################################
data = pd.read_pickle(PICKLE_PATH)
# print(data.isnull().sum())
# print(data['performance'].mean())

# TRANSFORMAR VALORS NULL #########################################################################################
for column in data.columns:
    if data[column].isnull().any():  # Verifica si la columna tiene valores nulos
        if data[column].dtype == 'object' or data[column].dtype.name == 'category':  # Columnas categóricas
            mode_value = data[column].mode()[0]  # Calcular la moda
            data[column] = data[column].fillna(mode_value)  # Reasignar a la columna
        elif np.issubdtype(data[column].dtype, np.number):  # Columnas numéricas
            mean_value = data[column].mean()  # Calcular la media
            data[column] = data[column].fillna(mean_value)  # Reasignar a la columna

# Confirmar que ya no hay valores nulos y ver cómo afectan los cambios realizados
# print(data.isnull().sum())
# print(data['performance'].mean())

# ELIMINAR DUPLICATS ##############################################################################################
res = data.drop_duplicates(inplace=True)
print(f'Hi havia {res} duplicats\n')

# CORREGIR ERRORS TIPOGRÀFICS ####################################################################################
# print(data.select_dtypes(include=['object', 'string']).columns)
for column in data.select_dtypes(include=['object', 'string']).columns:
    data[column] = data[column].str.lower().str.strip()  # Estandarditza

# CONVERTIR TIPUS DE DADES INCORRECTES ############################################################################
for i, tipo in data.dtypes.items():
    print(i, tipo)
# canviar = 

# NORMALITZACIÓ ###################################################################################################
# dades categòriques
# dades numèriques?

# GUARDEM EL DATASET NET ##########################################################################################
data.to_pickle(CLEANED_PICKLE_PATH)