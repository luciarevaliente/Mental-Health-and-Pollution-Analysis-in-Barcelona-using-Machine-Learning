"""
Script per netejar les dades de salut mental i contaminació de BCN
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 03/12/24
Descripció: Aquest script neteja les dades de salut mental i contaminació.
"""
# 03_data_cleaning.py
# IMPORTACIÓ
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# VARIABLES CONSTANTS
PICKLE_PATH = 'data/dataset.pkl'
CLEANED_PICKLE_PATH = 'data/cleaned_dataset.pkl'
k = 5  # KKNImputer
ESBORRAR = ['date_all', 'year', 'month', 'hour', 'day', 'start_year','start_month', 'start_day', 'start_hour', 'end_year','end_month', 'end_day', 'end_hour', 'Houron', 'Houroff', # Les dades temporals de l'enquesta no aporten informació, només per controlar el dataset
            'stroop_test', # Només pren un valor
            'yearbirth', # Ja tenim la variable age 
            'no2gps_12hx30','no2gps_24hx30','no2gps_12h','no2bcn_12h_x30','no2bcn_24h_x30','no2bcn_12h','no2gps_12h_x30','no2gps_24h_x30'  # Factors de 30
            ] 

# FUNCIONS ###############################################################
# Eliminar valors nuls
def filtrar_valors_null(dataset, k):
    data_cleaned = dataset.copy()
    data_cleaned = data_cleaned.loc[:, data_cleaned.isnull().mean() < 0.5]  # Eliminar columnes amb més del 50% de NaN

    knn_imputer = KNNImputer(n_neighbors=k)
    data_imputed = knn_imputer.fit_transform(data_cleaned.select_dtypes(include=['number']))
    data_cleaned[data_cleaned.select_dtypes(include=['number']).columns] = data_imputed

    for column in data_cleaned.select_dtypes(include=['object']).columns:
        mode_value = data_cleaned[column].mode()[0]
        data_cleaned[column] = data_cleaned[column].fillna(mode_value)
    return data_cleaned

# Eliminar duplicats
def eliminar_duplicats(dataset):
    return dataset.drop_duplicates()

# Convertir tipus de dades
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

# Normalitzar valors categòrics
def estandarditzar_valors_categorics(dataset, normalitzar=[]):
    for column in dataset.select_dtypes(include=['object', 'string']).columns:
        dataset[column] = dataset[column].str.lower().str.strip()
    for col in normalitzar:
        dataset[col] = dataset[col].round().astype('Int64')
    return dataset

# Escalar dades numèriques
def escalar_dades(dataset, numeric_columns):
    scaler = StandardScaler()
    dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])
    return dataset

# Codificar dades categòriques
def codificar_dades_categoriques(dataset, categorical_columns):
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(dataset[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=dataset.index)
    dataset = dataset.drop(columns=categorical_columns)
    dataset = pd.concat([dataset, encoded_df], axis=1)
    return dataset

# PROCESSAMENT ##########################################################
if __name__=="__main__":
    # Pas 0: Carregar dades i eliminar columnes innecessàries
    data = pd.read_pickle(PICKLE_PATH)
    data = data.drop(columns=ESBORRAR, errors='ignore')

    # Pas 1: Tractar valors nuls
    cleaned_dataset = filtrar_valors_null(data, k)

    # Pas 2: Eliminar duplicats
    cleaned_dataset = eliminar_duplicats(cleaned_dataset)

    # Pas 3: Convertir tipus de dades
    transform_to_int = ['occurrence_mental', 'occurrence_stroop', 'correct', 'response_duration_ms', 
                        'age_yrs', 'hour_gps', 'sec_noise55_day', 'sec_noise65_day', 'sec_greenblue_day',
                        'hours_noise_55_day', 'hours_noise_65_day', 'hours_greenblue_day', 'precip_12h_binary',
                        'precip_24h_binary', 'dayoftheweek', 'bienestar', 'energia', 
                        'estres', 'sueno']
    transform_to_str = ['mentalhealth_survey', 'ordenador', 'dieta', 'alcohol', 'drogas', 'enfermo', 'otrofactor', 
                        'district', 'education', 'access_greenbluespaces_300mbuff', 'smoke', 'psycho', 'gender', 
                        'Totaltime_estimated'] 
    cleaned_dataset = convertir_tipus_de_dades(cleaned_dataset, enter=transform_to_int, caracter=transform_to_str)

    # Pas 4: Normalitzar valors categòrics
    cleaned_dataset = estandarditzar_valors_categorics(cleaned_dataset, ['bienestar', 'energia', 'estres', 'sueno'])
    print(cleaned_dataset.shape)

    # GUARDAR EL DATASET ######################################################
    cleaned_dataset.to_pickle(CLEANED_PICKLE_PATH)

    # Exportar a Excel
    try:
        cleaned_dataset.to_excel('data/cleaned_dataset.xlsx', index=False)
        print(f"L'arxiu ha estat exportat a: 'data/cleaned_dataset.xlsx'")
    except Exception as e:
        print(f"Error: {e}")
