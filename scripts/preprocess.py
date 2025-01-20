"""
Script per preprocessar les dades de salut mental i contaminació de BCN
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 08/12/24
Descripció: Aquest script carrega, codifica i escala les dades de salut mental i contaminació.
"""
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import pandas as pd
import numpy as np

def codificar_columnas(dataset, ordinal_columns, binary_columns, nominal_columns):
    """
    Codifica les columnes categòriques del dataset:
    - OrdinalEncoder per a columnes ordinals, seguint l'ordre especificat.
    - Transforma valors de columnes binàries (p. ex., 'yes/no', 'hombre/mujer') a format numèric (1/-1).
    - OneHotEncoder per a columnes nominals, generant noves columnes binàries per a cada categoria.

    Parameters:
        dataset (DataFrame): Dataset a processar.
        ordinal_columns (dict): Columnes ordinals i el seu ordre.
        binary_columns (list): Columnes binàries.
        nominal_columns (list): Columnes nominals.

    Returns:
        dataset (DataFrame): Dataset amb les columnes codificades.
        ordinal_encoder (OrdinalEncoder): Encoder utilitzat per a columnes ordinals.
        encoded_nominals (numpy.ndarray): Valors codificats de columnes nominals.
    """
    # Codificar las columnas ordinales
    if ordinal_columns:
        print(f"Codificando columnas ordinales: {list(ordinal_columns.keys())}")
        ordinal_encoder = OrdinalEncoder(categories=list(ordinal_columns.values()))
        dataset[list(ordinal_columns.keys())] = ordinal_encoder.fit_transform(dataset[list(ordinal_columns.keys())])
    else:
        ordinal_encoder = None

    # Codificar las columnas binarias
    print(f"Codificando columnas binarias: {list(binary_columns)}")
    if binary_columns:
        for col in binary_columns:
            dataset[col] = dataset[col].map({'yes': 1, 'no': -1, 
                                            'hombre': 1, 'mujer': -1, 
                                            0: -1})

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
    else:
        encoded_nominals = None
    
    # Retornar el dataset modificado
    return dataset, ordinal_encoder, encoded_nominals

def escalar(dataset, numerical_columns):
    """
    Escala les columnes numèriques del dataset utilitzant StandardScaler (Z-score).

    Parameters:
        dataset (DataFrame): Dataset a escalar.
        numerical_columns (list): Columnes numèriques a escalar.

    Returns:
        scaler_scale (numpy.ndarray): Escales utilitzades.
        scaler_mean (numpy.ndarray): Mitjanes utilitzades.
    """
    if not numerical_columns.empty:
        # Aplicar escalado
        scaler = StandardScaler()
        dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])
    return scaler.scale_, scaler.mean_

def preprocess(CLEANED_DATASET_PATH, TARGET):
    """
    Preprocessa un dataset:
    - Càrrega el dataset.
    - Codifica columnes ordinals, binàries i nominals.
    - Escala columnes numèriques.
    - Barreja les files i guarda el dataset preprocessat.

    Parameters:
        CLEANED_DATASET_PATH (str): Ruta del dataset netejat.
        TARGET (str): Nom de la variable objectiu.

    Returns:
        data (DataFrame): Dataset preprocessat.
        scaler_scale, scaler_mean: Escales i mitjanes calculades.
        ordinal_encoder, nominal_encoder: Encoders utilitzats.
    """
    # Cargar el dataset
    initial_dataset = pd.read_pickle(CLEANED_DATASET_PATH)

    data = initial_dataset.drop(TARGET, axis=1)

    # Seleccionar columnas numéricas
    cols_int_to_change = ['precip_12h_binary', 'precip_24h_binary']  
    cols_int_to_change = list(set(cols_int_to_change).intersection(data.columns))  # Filtrar existentes
    
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.difference(cols_int_to_change)
    print(f"Numerical columns: {numerical_columns}")
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

    # Mantener solo las claves que existen en el dataset
    ordinal_columns = {col: order for col, order in ordinal_columns.items() if col in data.columns}

    # Separar columnas nominales y binarias
    binary_columns = ['mentalhealth_survey', 'Totaltime_estimated', 'access_greenbluespaces_300mbuff', 'actividadfisica', 'alcohol', 'bebida', 'dieta', 'drogas', 'enfermo', 'gender', 'ordenador', 'otrofactor', 'psycho', 'smoke']
    binary_columns = list(set(binary_columns).intersection(data.columns))  # Filtrar existentes
    
    nominal_columns = data.select_dtypes(include=['object']).columns.difference(ordinal_columns.keys()).difference(binary_columns)

    # Aplicar codificación
    data, ordinal_encoder, nominal_encoder = codificar_columnas(data, ordinal_columns, binary_columns, nominal_columns)

    # Codifiquem les columnes numèriques binàries que falten
    for col in cols_int_to_change:
        data[col] = np.where(data[col] == 0.0, -1, data[col])  # Aplicar modificació

    # Escalar datos numéricos
    scaler_scale, scaler_mean = escalar(data, numerical_columns)
    
    data[TARGET] = initial_dataset[TARGET]

    # Hacer shuffle de las filas después de escalar
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Guardar dataset escalado
    data.to_pickle('data/processed_dataset.pkl')
    print("Dataset escalado guardado.")
    
    return data, scaler_scale, scaler_mean, ordinal_encoder, nominal_encoder

def revert_preprocessing(centroides_df, scaler_scale, scaler_mean, ordinal_columns, binary_columns, nominal_columns, ordinal_encoder=None, nominal_encoder=None):
    """
    Reverteix el preprocessament d'un DataFrame, desescalant i descodificant.

    Parameters:
        centroides_df (DataFrame): DataFrame a revertir.
        scaler_scale, scaler_mean: Escales i mitjanes utilitzades.
        ordinal_columns, binary_columns, nominal_columns (list): Columnes afectades.
        ordinal_encoder, nominal_encoder: Encoders utilitzats.

    Returns:
        centroides_df (DataFrame): DataFrame amb valors originals.
    """
    # Verificar si hay columnas numéricas antes de desescalar
    if scaler_scale is not None and scaler_mean is not None:
        num_columns = [
            col for col in centroides_df.columns 
            if col not in ordinal_columns and col not in binary_columns and col not in nominal_columns
        ]
        
        if num_columns:
            print(f"Desescalando columnas numéricas: {num_columns}")

            # Verificar coincidencia entre dimensiones
            if len(num_columns) != len(scaler_scale) or len(num_columns) != len(scaler_mean):
                raise ValueError(
                    f"La cantidad de columnas numéricas ({len(num_columns)}) no coincide con las longitudes "
                    f"de scaler_scale ({len(scaler_scale)}) y scaler_mean ({len(scaler_mean)})."
                )

            # Desescalar las columnas numéricas
            centroides_df[num_columns] = centroides_df[num_columns] * scaler_scale + scaler_mean

    # Verificar si hay columnas binarias antes de descodificar
    if binary_columns:
        print(f"Revirtiendo columnas binarias: {binary_columns}")
        for col in binary_columns:
            # Convertir valores continuos a discretos (0 y 1)
            centroides_df[col] = (centroides_df[col] >= 0).astype(int)

    # Verificar si hay columnas ordinales y un encoder válido antes de descodificar
    if ordinal_columns and ordinal_encoder is not None:
        print(f"Revirtiendo columnas ordinales: {ordinal_columns}")
        centroides_df[ordinal_columns] = ordinal_encoder.inverse_transform(centroides_df[ordinal_columns])

    # Verificar si hay columnas nominales y un encoder válido antes de descodificar
    if nominal_columns and nominal_encoder is not None:
        print(f"Revirtiendo columnas nominales: {nominal_columns}")
        nominal_columns_encoded = [
            col for col in centroides_df.columns 
            if any(col.startswith(nom) for nom in nominal_columns)
        ]
        if nominal_columns_encoded:
            decoded_nominals = nominal_encoder.inverse_transform(centroides_df[nominal_columns_encoded])
            centroides_df[nominal_columns] = decoded_nominals
            centroides_df.drop(columns=nominal_columns_encoded, inplace=True)

    # Retornar el DataFrame modificado
    return centroides_df
