from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

CLEANED_DATASET_PATH = 'data/cleaned_dataset.pkl'

def codificar_columnas(dataset, ordinal_columns, binary_columns, nominal_columns):
    """
    Codifica las columnas categóricas:
    - OrdinalEncoder para columnas ordinales.
    - OneHotEncoder para columnas nominales.

    Args:
        dataset (DataFrame): Dataset con columnas categóricas y numéricas.

    Returns:
        DataFrame: Dataset con columnas categóricas codificadas.
    """
    # Codificar las columnas ordinales
    if ordinal_columns:
        print(f"Codificando columnas ordinales: {list(ordinal_columns.keys())}")
        ordinal_encoder = OrdinalEncoder(categories=list(ordinal_columns.values()))
        dataset[list(ordinal_columns.keys())] = ordinal_encoder.fit_transform(dataset[list(ordinal_columns.keys())])

    # Codificar las columnas binarias
    for col in binary_columns:
        dataset[col] = dataset[col].map({'yes':1, 'no':-1, 
                                         'hombre':1, 'mujer':-1})

    # Codificar las columnas nominales 
    if len(nominal_columns) > 0:
        print(f"Codificando columnas nominales: {list(nominal_columns)}")
        nominal_encoder = OneHotEncoder(sparse_output=False)
        encoded_nominals = nominal_encoder.fit_transform(dataset[nominal_columns])

        # Els valors que siguin 0, els cambien a -1
        encoded_nominals_zero = encoded_nominals==0
        encoded_nominals[encoded_nominals_zero] = -1
        
        # Convertir a DataFrame y concatenar con las columnas restantes
        encoded_df = pd.DataFrame(
            encoded_nominals,
            columns=nominal_encoder.get_feature_names_out(nominal_columns),
            index=dataset.index
        )
        dataset = pd.concat([dataset.drop(columns=nominal_columns), encoded_df], axis=1)

    return dataset

def escalar(dataset):
    """
    Función que escala los datos numéricos de un dataset utilizando StandardScaler (Estandarización, Z-score).

    StandardScaler estandariza las características para que tengan una media de 0 y una desviación estándar de 1. 
    Este tipo de escalado es útil si deseas que todas las características tengan un peso uniforme en el modelo, 
    sin que algunas sean más importantes simplemente debido a su escala.

    Args:
        dataset (DataFrame): Dataset con datos numéricos y categóricos.

    Returns:
        DataFrame: Dataset con datos numéricos escalados.
    """
    # Crear copia del dataset para no modificar el original
    dataset_scaled = dataset.copy()

    # Seleccionar columnas numéricas
    numerical_columns = dataset_scaled.select_dtypes(include=['float64', 'int64']).columns

    if numerical_columns.empty:
        print("No se encontraron columnas numéricas para escalar.")
        return dataset_scaled

    # Aplicar escalado
    scaler = StandardScaler()
    dataset_scaled[numerical_columns] = scaler.fit_transform(dataset_scaled[numerical_columns])

    print("Primeras filas de las columnas escaladas:")
    print(dataset_scaled[numerical_columns].head())

    return dataset_scaled

# Cargar el dataset
dataset_path = CLEANED_DATASET_PATH
data = pd.read_pickle(dataset_path)

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
nominal_columns = dataset.select_dtypes(include=['object']).columns.difference(ordinal_columns.keys())

binary_columns = ['mentalhealth_survey', 'Totaltime_estimated', 'access_greenbluespaces_300mbuff', 'actividadfisica', 'alcohol', 'bebida', 'dieta', 'drogas', 'enfermo', 'gender', 'ordenador', 'otrofactor', 'psycho', 'smoke']
nominal_columns = nominal_columns.difference(binary_columns)
    
# Aplicar codificación
dataset_codificado = codificar_columnas(data, ordinal_columns, binary_columns, nominal_columns)

# Guardar el resultado
dataset_codificado.to_pickle('data/codif_dataset.pkl')
dataset_codificado.to_excel('data/codif_dataset.xlsx', index=False)
print("Dataset codificado guardado.")
