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
col_str_real = ['dayoftheweek', 'mentalhealth_survey', 'bienestar', 'energia', 'estres', 'sueno', 'horasfuera', 'ordenador', 
           'dieta', 'alcohol', 'drogas', 'enfermo', 'otrofactor', 'district', 'education', 'access_greenbluespaces_300mbuff',
           'smoke', 'psycho', 'gender', 'stroop_test', 'Totaltime_estimated']

# print(data['stroop_test'])
# print(data['Totaltime_estimated'])
# print(data['Houron'])
# print(data['Houroff'])

# Convertir les columnes de dates a tipus datetime
for col in col_date_real:
    data[col] = pd.to_datetime(data[col], errors='coerce')  # 'coerce' per gestionar errors

# Convertir les columnes de hores a Format HH:MM
for col in col_hour_real:
    data[col] = pd.to_timedelta(data[col], errors='coerce')
    
# Convertir les columnes d'enteres a tipus numèric
for col in col_int_real:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # 'coerce' per substituir errors amb NaN

# Les columnes de tipus string (ja han de ser strings, però es poden netejar)
for col in col_str_real:
    data[col] = data[col].astype(str)  # Assegura't que són strings

# Comprovar si hi ha NaN en tot el DataFrame
nan_check = data.isna().any().any()  # Retorna True si hi ha qualsevol NaN al DataFrame

if nan_check:
    print("Hi ha valors NaN al DataFrame.")
    # Mostrar totes les files amb almenys un NaN
    nan_per_column = data.isna().sum()
    for i, v in nan_per_column.items():
        if v!=0:
            print(i,v)
    # print(nan_per_column)
else:
    print("No hi ha valors NaN al DataFrame.")

# CORREGIR ERRORS TIPOGRÀFICS ####################################################################################
# print(data.select_dtypes(include=['object', 'string']).columns)
for column in data.select_dtypes(include=['object', 'string']).columns:
    data[column] = data[column].str.lower().str.strip()  # Estandarditza

# GUARDEM EL DATASET NET ##########################################################################################
data.to_pickle(CLEANED_PICKLE_PATH)