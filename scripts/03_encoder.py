from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# VARIABLES
CLEANED_PICKLE_PATH = 'data/cleaned_dataset.pkl'
CODIF_PICKLE_PATH = 'data/codif_dataset.pkl'

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

def codificar_columnas(dataset):
    """
    Codifica las columnas categóricas:
    - OrdinalEncoder para columnas ordinales.
    - OneHotEncoder para columnas nominales.

    Args:
        dataset (DataFrame): Dataset con columnas categóricas y numéricas.

    Returns:
        DataFrame: Dataset con columnas categóricas codificadas.
    """
    # Definir las columnas ordinales y sus órdenes
    ordinal_columns = {
        "education": ["Primario o menos", "Bachillerato", "Universitario"],
        "covid_work": ["Ha empeorado mucho", "Ha empeorado un poco", "No ha cambiado", "Ha mejorado un poco", "Ha mejorado mucho"],
        "covid_mood": ["Ha empeorado mucho", "Ha empeorado un poco", "No ha cambiado", "Ha mejorado un poco", "Ha mejorado mucho"],
        "covid_sleep": ["Ha empeorado mucho", "Ha empeorado un poco", "No ha cambiado", "Ha mejorado un poco", "Ha mejorado mucho"],
        "covid_espacios": ["Le doy menos importancia que antes", "No ha cambiado", "Le doy más importancia que antes"],
        "covid_aire": ["Le doy menos importancia que antes", "No ha cambiado", "Le doy más importancia que antes"],
        "covid_motor": ["Lo utilizo menos que antes", "Lo utilizo igual que antes", "Lo utilizo más que antes"],
        "covid_electric": ["Lo utilizo menos que antes", "Lo utilizo igual que antes", "Lo utilizo más que antes"],
        "covid_bikewalk": ["Lo utilizo menos que antes", "Lo utilizo igual que antes", "Lo utilizo más que antes"],
        "covid_public_trans": ["Lo utilizo menos que antes", "Lo utilizo igual que antes", "Lo utilizo más que antes"]
    }

    # Separar columnas nominales
    nominal_columns = dataset.select_dtypes(include=['object']).columns.difference(ordinal_columns.keys())

    # Codificar las columnas ordinales
    if ordinal_columns:
        print(f"Codificando columnas ordinales: {list(ordinal_columns.keys())}")
        ordinal_encoder = OrdinalEncoder(categories=list(ordinal_columns.values()))
        dataset[list(ordinal_columns.keys())] = ordinal_encoder.fit_transform(dataset[list(ordinal_columns.keys())])

    # Codificar las columnas nominales
    if len(nominal_columns) > 0:
        print(f"Codificando columnas nominales: {list(nominal_columns)}")
        nominal_encoder = OneHotEncoder(sparse_output=False)
        encoded_nominals = nominal_encoder.fit_transform(dataset[nominal_columns])

        # Convertir a DataFrame y concatenar con las columnas restantes
        encoded_df = pd.DataFrame(
            encoded_nominals,
            columns=nominal_encoder.get_feature_names_out(nominal_columns),
            index=dataset.index
        )
        dataset = pd.concat([dataset.drop(columns=nominal_columns), encoded_df], axis=1)

    return dataset

# Cargar el dataset
dataset_path = '/mnt/data/CitieSHealth_BCN_DATA_PanelStudy_20220414.csv'
data = pd.read_csv(dataset_path)

# Aplicar codificación
dataset_codificado = codificar_columnas(data)

# Guardar el resultado
dataset_codificado.to_pickle('data/codif_dataset.pkl')
dataset_codificado.to_excel('data/codif_dataset.xlsx', index=False)
print("Dataset codificado guardado.")

