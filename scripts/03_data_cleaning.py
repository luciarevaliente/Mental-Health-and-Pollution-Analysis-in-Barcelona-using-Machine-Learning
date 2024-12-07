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
from sklearn.preprocessing import StandardScaler

# # VARIABLES CONSTANTS
# PICKLE_PATH = 'data/dataset.pkl'
# CLEANED_PICKLE_PATH = 'data/cleaned_dataset.pkl'
# k = 5
# ESBORRAR = ['date_all']

# # CARREGAR EL DATASET #############################################################################################
# # Carreguem el pickle del dataset original
# data = pd.read_pickle(PICKLE_PATH)

# # Eliminació de característiques no necessaries
# for col in ESBORRAR:
#     data.drop(col, axis='columns')

# # TRANSFORMAR VALORS NULL #########################################################################################
# def filtrar_valors_null(dataset, k):
#     """
#     Funció que elimina o transforma els valors nulls d'un dataset. Per cada columna, si té més d'un 50% de valors null, l'elimina. 
#     Si en té menys d'un 50%, imputa els valors faltants mitjançant els k veïns més propers.

#     Args:
#         dataset (DataFrame): Dataset a natejar.
#         k (int): K veïns més propers.

#     Returns:
#         cleaned_dataset: Dataset amb els valors null natejats.
#     """
#     # Copia del DataFrame
#     data_cleaned = data.copy()

#     # Eliminar columnas con demasiados valores faltantes (más del 50% de NaN)
#     data_cleaned = data_cleaned.loc[:, data_cleaned.isnull().mean() < 0.5]

#     # Usar KNNImputer para imputar valores numéricos
#     knn_imputer = KNNImputer(n_neighbors=k)
#     data_imputed = knn_imputer.fit_transform(data_cleaned.select_dtypes(include=['number']))

#     # Actualizar el DataFrame con los valores imputados
#     data_cleaned[data_cleaned.select_dtypes(include=['number']).columns] = data_imputed

#     # Imputar columnas categóricas con la moda
#     for column in data_cleaned.select_dtypes(include=['object']).columns:
#         mode_value = data_cleaned[column].mode()[0]  # Obtener la moda
#         data_cleaned[column] = data_cleaned[column].fillna(mode_value)  # Rellenar los NaN con la moda 
#     return data_cleaned

# cleaned_dataset = filtrar_valors_null(data, k)

# # Resultado después de la imputación
# print(f'Dataset con valores nulos imputados: {cleaned_dataset.isnull().any().sum()}')

# # ELIMINAR DUPLICATS ##############################################################################################
# def eliminacio_files_duplicades(dataset):
#     return dataset.drop_duplicates(inplace=True)

# res = eliminacio_files_duplicades(cleaned_dataset)
# print(f'\nHi havia {res} registres duplicats\n')

# # CONVERTIR TIPUS DE DADES INCORRECTES ############################################################################
# def convertir_tipus_de_dades(dataset, datetime=[], hora=[], enter=[], caracter=[]):
#     """Modifica el dataset que es passa com a paràmetre"""
#     # if datetime:
#     #     for col in datetime:
#     #         dataset[col] = pd.to_datetime(dataset[col], errors='coerce')  # 'coerce' per gestionar errors
#     # if hora:
#     #     for col in hora:
#     #         dataset[col] = pd.to_timedelta(dataset[col], errors='coerce')
#     if enter:
#         for col in enter:
#             dataset[col] = pd.to_numeric(dataset[col], errors='coerce').astype(int)  # 'coerce' per substituir errors amb NaN
#     if caracter:
#         for col in caracter:
#             dataset[col] = dataset[col].astype(str)  # Assegura't que són strings

# transform_to_int = ['occurrence_mental', 'occurrence_stroop', 'correct', 'response_duration_ms', 'start_day',
#             'start_month', 'start_year', 'start_hour', 'end_day', 'end_month', 'end_year', 'end_hour', 'age_yrs', 'yearbirth', 'hour_gps', 'sec_noise55_day',
#             'sec_noise65_day', 'sec_greenblue_day', 'hours_noise_55_day', 'hours_noise_65_day', 'hours_greenblue_day', 'precip_12h_binary',
#             'precip_24h_binary', 'dayoftheweek', 'year', 'month', 'day','hour', 'bienestar', 'energia', 'estres', 'sueno', 'occurrence_stroop', 'correct']
# transform_to_str = ['mentalhealth_survey',  'ordenador', 'dieta', 'alcohol', 'drogas', 'enfermo', 'otrofactor', 'district', 'education', 'access_greenbluespaces_300mbuff',
#            'smoke', 'psycho', 'gender', 'stroop_test', 'Totaltime_estimated']

# convertir_tipus_de_dades(cleaned_dataset, enter=transform_to_int, caracter=transform_to_str)

# # for i, v in cleaned_dataset.dtypes.items():
# #     print(i, v)
# # exit

# # Comprovar si hi ha NaN en tot el DataFrame
# nan_check = cleaned_dataset.isna().any().any()  # Retorna True si hi ha qualsevol NaN al DataFrame

# if nan_check:
#     print("Hi ha valors NaN al DataFrame.")
#     # Mostrar totes les files amb almenys un NaN
#     nan_per_column = cleaned_dataset.isna().sum()
#     for i, v in nan_per_column.items():
#         if v!=0:
#             print(i,v)
#     # print(nan_per_column)
# else:
#     print("No hi ha valors NaN al DataFrame.")



# # Transformació característiques d'hora
# # Comprobar si hay valores no finitos
# print(data['Houron'].isnull().sum())  # Verificar si hay NaNs
# print((data['Houron'] == float('inf')).sum())  # Verificar si hay infinitos
# print((data['Houron'] == float('-inf')).sum())  # Verificar si hay infinitos negativos
# print((data['Houron'] == 0).sum())  # Verificar si hay ceros

# # Convertir las columnas 'Houron' y 'Houroff' a tipo datetime
# data['Houron'] = pd.to_datetime(data['Houron'])
# data['Houroff'] = pd.to_datetime(data['Houroff'])
# print('convertit')

# # Extraer hora y minuto de 'Houron'
# data['Houron_hour'] = data['Houron'].dt.hour.astype(int)
# data['Houron_minute'] = data['Houron'].dt.minute.astype(int)
# print('houron hecho')

# # Extraer hora y minuto de 'Houroff'
# data['Houroff_hour'] = data['Houroff'].dt.hour.astype(int)
# data['Houroff_minute'] = data['Houroff'].dt.minute.astype(int)
# print('houroff hecho')

# data.drop(columns=['Houron', 'Houroff'])
# print('eliminado')
# exit


# # CORREGIR ERRORS TIPOGRÀFICS i NORMALITZACIÓ DADES CATEGÒRIQUES ####################################################
# for column in cleaned_dataset.select_dtypes(include=['object', 'string']).columns:
#     cleaned_dataset[column] = cleaned_dataset[column].str.lower().str.strip()  # Estandarditza
#     # print(cleaned_dataset[column].value_counts())
#     # print()
    
# # Valors a normalitzar
# normalitzar = ['bienestar', 'energia', 'estres', 'sueno']  #Estan com str(float) però són valors enters
# for col in normalitzar:
#     # cleaned_dataset[col] = cleaned_dataset[col].astype(float)
#     cleaned_dataset[col] = cleaned_dataset[col].round().astype(int)
#     # cleaned_dataset[col] = cleaned_dataset[col].astype(str) #  ??????????????????????????????
#     # print(cleaned_dataset[col].value_counts())
#     # print()

# # GUARDEM EL DATASET NET ##########################################################################################
# cleaned_dataset.to_pickle(CLEANED_PICKLE_PATH)

# # GUARDEM PICKLE COM A EXCEL PER COMPROVAR QUE TOT ÉS CORRECTE ####################################################
# # data_pickle = pd.read_pickle(CLEANED_PICKLE_PATH)

# # output_path = 'data/cleaned_dataset.xlsx'  # Substitueix per la ruta de sortida
# # cleaned_dataset.to_excel(output_path, index=False)

# # print(f"Fitxer Excel guardat a {output_path}")


import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# VARIABLES CONSTANTES
PICKLE_PATH = 'data/dataset.pkl'
CLEANED_PICKLE_PATH = 'data/cleaned_dataset.pkl'
k = 5
ESBORRAR = ['date_all']

# FUNCIONES ###############################################################
# Eliminar valores nulos
def filtrar_valors_null(dataset, k):
    data_cleaned = dataset.copy()
    data_cleaned = data_cleaned.loc[:, data_cleaned.isnull().mean() < 0.5]  # Eliminar columnas con más del 50% de NaN

    knn_imputer = KNNImputer(n_neighbors=k)
    data_imputed = knn_imputer.fit_transform(data_cleaned.select_dtypes(include=['number']))
    data_cleaned[data_cleaned.select_dtypes(include=['number']).columns] = data_imputed

    for column in data_cleaned.select_dtypes(include=['object']).columns:
        mode_value = data_cleaned[column].mode()[0]
        data_cleaned[column] = data_cleaned[column].fillna(mode_value)
    return data_cleaned

# Eliminar duplicados
def eliminar_duplicados(dataset):
    return dataset.drop_duplicates()

# Convertir tipos de datos
def convertir_tipus_de_dades(dataset, enter=[], caracter=[]):
    if enter:
        for col in enter:
            if col in dataset.columns:
                dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
                dataset[col] = dataset[col].round().fillna(0).astype('Int64')
    if caracter:
        for col in caracter:
            if col in dataset.columns:
                dataset[col] = dataset[col].astype(str)
    return dataset

# Normalizar valores categóricos
def estandarditzar_valors_categorics(dataset, normalitzar=[]):
    for column in dataset.select_dtypes(include=['object', 'string']).columns:
        dataset[column] = dataset[column].str.lower().str.strip()
    for col in normalitzar:
        dataset[col] = dataset[col].round().astype('Int64')
    return dataset

# Tratar outliers
def tractar_outliers(dataset, numeric_columns):
    for col in numeric_columns:
        if col in dataset.columns:
            Q1 = dataset[col].quantile(0.25)
            Q3 = dataset[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]
    return dataset

# Escalar datos numéricos
def escalar_dades(dataset, numeric_columns):
    scaler = StandardScaler()
    dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])
    return dataset

# Codificar datos categóricos
def codificar_dades_categoriques(dataset, categorical_columns):
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(dataset[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=dataset.index)
    dataset = dataset.drop(columns=categorical_columns)
    dataset = pd.concat([dataset, encoded_df], axis=1)
    return dataset

# PROCESAMIENTO ##########################################################
if __name__=="__main__":
    # Cargar datos
    data = pd.read_pickle(PICKLE_PATH)

    # Eliminar columnas innecesarias
    data = data.drop(columns=ESBORRAR, errors='ignore')

    # Conversión de formatos
    data['Houron'] = pd.to_datetime(data['Houron'], format='%H:%M:%S', errors='coerce')
    data['Houroff'] = pd.to_datetime(data['Houroff'], format='%H:%M:%S', errors='coerce')
    data['Houron_hour'] = data['Houron'].dt.hour.astype('Int64')
    data['Houron_minute'] = data['Houron'].dt.minute.astype('Int64')
    data['Houroff_hour'] = data['Houroff'].dt.hour.astype('Int64')
    data['Houroff_minute'] = data['Houroff'].dt.minute.astype('Int64')
    data = data.drop(columns=['Houron', 'Houroff'], errors='ignore')

    # Paso 1: Tratar valores nulos
    cleaned_dataset = filtrar_valors_null(data, k)

    # Paso 2: Eliminar duplicados
    cleaned_dataset = eliminar_duplicados(cleaned_dataset)

    # Paso 3: Convertir tipos de datos
    transform_to_int = ['occurrence_mental', 'occurrence_stroop', 'correct', 'response_duration_ms', 'start_day',
                        'start_month', 'start_year', 'start_hour', 'end_day', 'end_month', 'end_year', 'end_hour', 
                        'age_yrs', 'yearbirth', 'hour_gps', 'sec_noise55_day', 'sec_noise65_day', 'sec_greenblue_day',
                        'hours_noise_55_day', 'hours_noise_65_day', 'hours_greenblue_day', 'precip_12h_binary',
                        'precip_24h_binary', 'dayoftheweek', 'year', 'month', 'day', 'hour', 'bienestar', 'energia', 
                        'estres', 'sueno']
    transform_to_str = ['mentalhealth_survey', 'ordenador', 'dieta', 'alcohol', 'drogas', 'enfermo', 'otrofactor', 
                        'district', 'education', 'access_greenbluespaces_300mbuff', 'smoke', 'psycho', 'gender', 
                        'stroop_test', 'Totaltime_estimated']
    cleaned_dataset = convertir_tipus_de_dades(cleaned_dataset, enter=transform_to_int, caracter=transform_to_str)

    # Paso 4: Normalizar valores categóricos
    cleaned_dataset = estandarditzar_valors_categorics(cleaned_dataset, ['bienestar', 'energia', 'estres', 'sueno'])

    # Paso 5: Tratar outliers
    numeric_columns = cleaned_dataset.select_dtypes(include=['number']).columns
    cleaned_dataset = tractar_outliers(cleaned_dataset, numeric_columns)

    # Paso 6: Escalar datos numéricos
    # cleaned_dataset = escalar_dades(cleaned_dataset, numeric_columns) # DEPENDE!!!!!!!!!!!!!!!!!!!!!!!!

    # Paso 7: Codificar datos categóricos
    # categorical_columns = cleaned_dataset.select_dtypes(include=['object', 'string']).columns
    # cleaned_dataset = codificar_dades_categoriques(cleaned_dataset, categorical_columns)

    # GUARDAR EL DATASET ######################################################
    cleaned_dataset.to_pickle(CLEANED_PICKLE_PATH)

    # Exportar a Excel
    # try:
    #     cleaned_dataset.to_excel('data/cleaned_dataset.xlsx', index=False)
    #     print(f"El archivo ha sido exportado a: 'data/cleaned_dataset.xlsx'")
    # except Exception as e:
    #     print(f"Error: {e}")
