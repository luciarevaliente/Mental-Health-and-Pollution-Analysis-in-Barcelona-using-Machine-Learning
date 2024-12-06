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
from sklearn.impute import KNNImputer

# VARIABLES CONSTANTS
PICKLE_PATH = 'data/dataset.pkl'
CLEANED_PICKLE_PATH = 'data/cleaned_dataset.pkl'

# CARREGAR EL DATASET #############################################################################################
# Carreguem el pickle del dataset original
data = pd.read_pickle(PICKLE_PATH)


# TRANSFORMAR VALORS NULL #########################################################################################
def filtrar_valors_null(dataset, k):
    """
    Funció que elimina o transforma els valors nulls d'un dataset. Per cada columna, si té més d'un 50% de valors null, l'elimina. 
    Si en té menys d'un 50%, imputa els valors faltants mitjançant els k veïns més propers.

    Args:
        dataset (DataFrame): Dataset a natejar.
        k (int): K veïns més propers.

    Returns:
        cleaned_dataset: Dataset amb els valors null natejats.
    """
    # Copia del DataFrame
    data_cleaned = data.copy()

    # Eliminar columnas con demasiados valores faltantes (más del 50% de NaN)
    data_cleaned = data_cleaned.loc[:, data_cleaned.isnull().mean() < 0.5]

    # Usar KNNImputer para imputar valores numéricos
    knn_imputer = KNNImputer(n_neighbors=5)
    data_imputed = knn_imputer.fit_transform(data_cleaned.select_dtypes(include=['number']))

    # Actualizar el DataFrame con los valores imputados
    data_cleaned[data_cleaned.select_dtypes(include=['number']).columns] = data_imputed

    # Imputar columnas categóricas con la moda
    for column in data_cleaned.select_dtypes(include=['object']).columns:
        data_cleaned[column].fillna(data_cleaned[column].mode()[0], inplace=True)

    return data_cleaned

cleaned_dataset = filtrar_valors_null(data)

# Resultado después de la imputación
print("Dataset con valores nulos imputados:")
print(cleaned_dataset.head())

# ELIMINAR DUPLICATS ##############################################################################################
res = cleaned_dataset.drop_duplicates(inplace=True)
print(f'Hi havia {res} registres duplicats\n')

# CONVERTIR TIPUS DE DADES INCORRECTES ############################################################################
# for i, tipo in data.dtypes.items():
#     print(i, tipo)

col_date_docu = ['date_all']
col_int_docu = ['occurrence_mental', 'stroop_test', 'occurrence_stroop', 'correct', 'response_duration_ms', 'start_day',
            'start_month', 'start_year', 'start_hour', 'end_day', 'end_month', 'end_year', 'end_hour', 
            'Totaltime', 'Totaltime_estimated', 'Houron', 'Houroff', 'age_yrs', 'yearbirth', 'hour_gps', 'sec_noise55_day',
            'sec_noise65_day', 'sec_greenblue_day', 'hours_noise_55_day', 'hours_noise_65_day', 'hours_greenblue_day', 'precip_12h_binary',
            'precip_24h_binary']
col_str_docu = ['dayoftheweek', 'mentalhealth_survey', 'bienestar', 'energia', 'estres', 'sueno', 'horasfuera', 'ordenador', 
           'dieta', 'alcohol', 'drogas', 'enfermo', 'otrofactor', 'district', 'education', 'access_greenbluespaces_300mbuff',
           'smoke', 'psycho', 'gender']

col_date_real = ['date_all']
col_hour_real = ['Houron', 'Houroff']
col_int_real = ['occurrence_mental', 'occurrence_stroop', 'correct', 'response_duration_ms', 'start_day',
            'start_month', 'start_year', 'start_hour', 'end_day', 'end_month', 'end_year', 'end_hour', 
            'Totaltime', 'Houron', 'Houroff', 'age_yrs', 'yearbirth', 'hour_gps', 'sec_noise55_day',
            'sec_noise65_day', 'sec_greenblue_day', 'hours_noise_55_day', 'hours_noise_65_day', 'hours_greenblue_day', 'precip_12h_binary',
            'precip_24h_binary']
col_str_real = ['dayoftheweek', 'mentalhealth_survey', 'bienestar', 'energia', 'estres', 'sueno', 'ordenador', 
           'dieta', 'alcohol', 'drogas', 'enfermo', 'otrofactor', 'district', 'education', 'access_greenbluespaces_300mbuff',
           'smoke', 'psycho', 'gender', 'stroop_test', 'Totaltime_estimated']

# print(data['stroop_test'])
# print(data['Totaltime_estimated'])
# print(data['Houron'])
# print(data['Houroff'])

# Convertir les columnes de dates a tipus datetime
for col in col_date_real:
    cleaned_dataset[col] = pd.to_datetime(cleaned_dataset[col], errors='coerce')  # 'coerce' per gestionar errors

# Convertir les columnes de hores a Format HH:MM
for col in col_hour_real:
    cleaned_dataset[col] = pd.to_timedelta(cleaned_dataset[col], errors='coerce')
    
# Convertir les columnes d'enteres a tipus numèric
for col in col_int_real:
    cleaned_dataset[col] = pd.to_numeric(cleaned_dataset[col], errors='coerce')  # 'coerce' per substituir errors amb NaN

# Les columnes de tipus string (ja han de ser strings, però es poden netejar)
for col in col_str_real:
    cleaned_dataset[col] = cleaned_dataset[col].astype(str)  # Assegura't que són strings

# Comprovar si hi ha NaN en tot el DataFrame
nan_check = cleaned_dataset.isna().any().any()  # Retorna True si hi ha qualsevol NaN al DataFrame

if nan_check:
    print("Hi ha valors NaN al DataFrame.")
    # Mostrar totes les files amb almenys un NaN
    nan_per_column = cleaned_dataset.isna().sum()
    for i, v in nan_per_column.items():
        if v!=0:
            print(i,v)
    # print(nan_per_column)
else:
    print("No hi ha valors NaN al DataFrame.")


# CORREGIR ERRORS TIPOGRÀFICS i NORMALITZACIÓ DADES CATEGÒRIQUES ####################################################
for column in cleaned_dataset.select_dtypes(include=['object', 'string']).columns:
    cleaned_dataset[column] = cleaned_dataset[column].str.lower().str.strip()  # Estandarditza
    print(cleaned_dataset[column].value_counts())
    print()
    

# Valors a normalitzar
normalitzar = ['bienestar', 'energia', 'estres', 'sueno']  #Estan com str(float) però són valors enters
for col in normalitzar:
    cleaned_dataset[col] = cleaned_dataset[col].astype(float)
    cleaned_dataset[col] = cleaned_dataset[col].round().astype(int)
    cleaned_dataset[col] = cleaned_dataset[col].astype(str) #  ??????????????????????????????
    # print(cleaned_dataset[col].value_counts())
    # print()

# print(cleaned_dataset['sueno'].dtype)
# Como hay float en el dataset hay que eliminarlos de aquí!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# GUARDEM EL DATASET NET ##########################################################################################
cleaned_dataset.to_pickle(CLEANED_PICKLE_PATH)

# GUARDEM PICKLE COM A EXCEL PER COMPROVAR QUE TOT ÉS CORRECTE ####################################################
# data_pickle = pd.read_pickle(CLEANED_PICKLE_PATH)

# output_path = 'data/cleaned_dataset.xlsx'  # Substitueix per la ruta de sortida
# data.to_excel(output_path, index=False)

# print(f"Fitxer Excel guardat a {output_path}")
