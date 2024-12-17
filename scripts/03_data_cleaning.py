"""
Script per natejar les dades de salut mental i contaminació de BCN
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 03/12/24
Descripció: Aquest script nateja les dades de salut mental i contaminació.
"""
# 03_data_cleaning.py
# IMPORTACIÓ
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# VARIABLES CONSTANTES
PICKLE_PATH = 'data/dataset.pkl'
CLEANED_PICKLE_PATH = 'data/cleaned_dataset.pkl'
k = 5
ESBORRAR = ['date_all', 'year', 'month', 'hour', 'day', 'start_year','start_month', 'start_day', 'start_hour', 'end_year','end_month', 'end_day', 'end_hour', 'Houron', 'Houroff', # Les dades temporals de l'enquesta no aporten info, només per controlar dataset
            'stroop_test', # Només pren un valor
            'yearbirth'] # Ja tenim la variable age 

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

    # # Conversión de formatos
    # data['Hour] = pd.to_datetime(data['Houron'], format='%H:%M:%S', errors='coerce')
    # data['Houroff'] = pd.to_datetime(data['Houroff'], format='%H:%M:%S', errors='coerce')
    # data['Houron_hour'] = data['Houron'].dt.hour.astype('Int64')
    # data['Houron_minute'] = data['Houron'].dt.minute.astype('Int64')
    # data['Houroff_hour'] = data['Houroff'].dt.hour.astype('Int64')
    # data['Houroff_minute'] = data['Houroff'].dt.minute.astype('Int64')
    # data = data.drop(columns=['Houron', 'Houroff'], errors='ignore')on'

    # Paso 1: Tratar valores nulos
    cleaned_dataset = filtrar_valors_null(data, k)

    # Paso 2: Eliminar duplicados
    cleaned_dataset = eliminar_duplicados(cleaned_dataset)

    # Paso 3: Convertir tipos de datos
    transform_to_int = ['occurrence_mental', 'occurrence_stroop', 'correct', 'response_duration_ms', 
                        'age_yrs', 'hour_gps', 'sec_noise55_day', 'sec_noise65_day', 'sec_greenblue_day',
                        'hours_noise_55_day', 'hours_noise_65_day', 'hours_greenblue_day', 'precip_12h_binary',
                        'precip_24h_binary', 'dayoftheweek', 'bienestar', 'energia', 
                        'estres', 'sueno']
    transform_to_str = ['mentalhealth_survey', 'ordenador', 'dieta', 'alcohol', 'drogas', 'enfermo', 'otrofactor', 
                        'district', 'education', 'access_greenbluespaces_300mbuff', 'smoke', 'psycho', 'gender', 
                        'Totaltime_estimated'] 
    cleaned_dataset = convertir_tipus_de_dades(cleaned_dataset, enter=transform_to_int, caracter=transform_to_str)

    # Paso 4: Normalizar valores categóricos
    cleaned_dataset = estandarditzar_valors_categorics(cleaned_dataset, ['bienestar', 'energia', 'estres', 'sueno'])

    # Paso 5: Tratar outliers
    numeric_columns = cleaned_dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cleaned_dataset = tractar_outliers(cleaned_dataset, numeric_columns)

    # GUARDAR EL DATASET ######################################################
    cleaned_dataset.to_pickle(CLEANED_PICKLE_PATH)

    # Exportar a Excel
    try:
        cleaned_dataset.to_excel('data/cleaned_dataset.xlsx', index=False)
        print(f"El archivo ha sido exportado a: 'data/cleaned_dataset.xlsx'")
    except Exception as e:
        print(f"Error: {e}")
