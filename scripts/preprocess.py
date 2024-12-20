"""
Script per preprocessar les dades de salut mental i contaminació de BCN
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 08/12/24
Descripció: Aquest script carrega, codifica i escala les dades de salut mental i contaminació.
"""
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import pandas as pd
import numpy as np

CLEANED_DATASET_PATH = 'data/cleaned_dataset.pkl'
TARGET = 'estres'

def codificar_columnas(dataset, ordinal_columns, binary_columns, nominal_columns):
    """
    Codifica las columnas categóricas:
    - OrdinalEncoder para columnas ordinales.
    - OneHotEncoder para columnas nominales.
    """
    # Codificar las columnas ordinales
    if ordinal_columns:
        print(f"Codificando columnas ordinales: {list(ordinal_columns.keys())}")
        ordinal_encoder = OrdinalEncoder(categories=list(ordinal_columns.values()))
        dataset[list(ordinal_columns.keys())] = ordinal_encoder.fit_transform(dataset[list(ordinal_columns.keys())])

    # Codificar las columnas binarias
    print(f"Codificando columnas binarias: {list(binary_columns)}")
    for col in binary_columns:
        dataset[col] = dataset[col].map({'yes': 1, 'no': -1, 'hombre': 1, 'mujer': -1, 0: -1})

    # Codificar las columnas nominales
    if len(nominal_columns) > 0:
        print(f"Codificando columnas nominales: {list(nominal_columns)}")
        
        # Inicializamos el OneHotEncoder
        nominal_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # Ajustamos y transformamos las columnas nominales
        encoded_nominals = nominal_encoder.fit_transform(dataset[nominal_columns])
        zeros = encoded_nominals == 0
        encoded_nominals[zeros] = -1
        
        # Crear DataFrame con las columnas codificadas
        encoded_df = pd.DataFrame(
            encoded_nominals,
            columns=nominal_encoder.get_feature_names_out(nominal_columns),
            index=dataset.index
        )


        # Concatenamos el DataFrame codificado con el original (sin las columnas nominales originales)
        dataset = pd.concat([dataset.drop(columns=nominal_columns), encoded_df], axis=1)
    
    # Retornar el dataset modificado
    return dataset

def escalar(dataset, numerical_columns):
    """
    Función que escala los datos numéricos de un dataset utilizando StandardScaler (Estandarización, Z-score).
    """
    if not numerical_columns.empty:
        # Aplicar escalado
        scaler = StandardScaler()
        dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

def preprocess(CLEANED_DATASET_PATH, TARGET):
    """Li passem el dataset a codificar i escalar i la variable target"""
    # Cargar el dataset
    initial_dataset = pd.read_pickle(CLEANED_DATASET_PATH)
   
    data = initial_dataset.drop(TARGET, axis=1)

    # Seleccionar columnas numéricas
    cols_int_to_change = ['precip_12h_binary', 'precip_24h_binary']  # Eliminem pq son binàries de tipus int --> excepció
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.difference(cols_int_to_change)

    # Definir las columnas ordinales y sus órdenes
    ordinal_columns = {
        "education": ["primario o menos", "bachillerato", "universitario"],
        "covid_work": ["ha empeorado mucho", "ha empeorado un poco", "no ha cambiado", "ha mejorado un poco", "ha mejorado mucho"],
        "covid_mood": ["ha empeorado mucho", "ha empeorado un poco", "no ha cambiado", "ha mejorado un poco", "ha mejorado mucho"],
        "covid_sleep": ["ha empeorado mucho", "ha empeorado un poco", "no ha cambiado", "ha mejorado un poco", "ha mejorado mucho"],
        "covid_espacios": ["le doy menos importancia que antes", "no ha cambiado", "le doy más importancia que antes"],
        "covid_aire": ["le doy menos importancia que antes", "no ha cambiado", "le doy más importancia que antes"],
        "covid_motor": ["lo utilizo menos que antes", "lo utilizo igual que antes", "lo utilizo más que antes"],
        "covid_electric": ["lo utilizo menos que antes", "lo utilizo igual que antes", "lo utilizo más que antes"],
        "covid_bikewalk": ["lo utilizo menos que antes", "lo utilizo igual que antes", "lo utilizo más que antes"],
        "covid_public_trans": ["lo utilizo menos que antes", "lo utilizo igual que antes", "lo utilizo más que antes"]
    }

    # Separar columnas nominales y binarias
    binary_columns = ['mentalhealth_survey', 'Totaltime_estimated', 'access_greenbluespaces_300mbuff', 'actividadfisica', 'alcohol', 'bebida', 'dieta', 'drogas', 'enfermo', 'gender', 'ordenador', 'otrofactor', 'psycho', 'smoke']
    nominal_columns = data.select_dtypes(include=['object']).columns.difference(ordinal_columns.keys()).difference(binary_columns)

    # Aplicar codificación
    data = codificar_columnas(data, ordinal_columns, binary_columns, nominal_columns)

    # Codifiquem les columnes numèriques binàries que falten
    for col in cols_int_to_change:
        data[col] = np.where(data[col] == 0.0, -1, data[col])  # Aplicar modificació

    # Guardar el resultado
    # data.to_pickle('data/codif_dataset.pkl')
    # data.to_excel('data/codif_dataset.xlsx', index=False)
    # print("Dataset codificado guardado.")

    # Escalar datos numéricos
    escalar(data, numerical_columns)
    
    data[TARGET] = initial_dataset[TARGET]

    # Hacer shuffle de las filas después de escalar
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Guardar dataset escalado
    # data.to_pickle('data/shuffled_scaled_dataset.pkl')
    # data.to_excel('data/shuffled_scaled_dataset.xlsx', index=False)
    # print("Dataset escalado guardado.")
    
    return data


if __name__=="__main__":
    preprocess(CLEANED_DATASET_PATH, TARGET)