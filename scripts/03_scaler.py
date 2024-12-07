# IMPORTACIÓ
from sklearn.preprocessing import StandardScaler
import pandas as pd

# VARIABLES
PICKLE_PATH = 'data/cleaned_dataset.pkl'
OUT_PICKLE_PATH = 'data/scaled_dataset.pkl'

# ESCALAT DE DADES NUMÈRIQUES ####################################################################################
def escalar(dataset):
    """
    Función que escala los datos numèricos de un dataset utilizando StandardScaler (Estandarización, Z-score).

    StandardScaler estandariza las características para que tengan una media de 0 y una desviación estándar de 1. 
    Este tipo de escalado es útil si deseas que todas las características tengan un peso uniforme en el modelo, 
    sin que algunas sean más importantes simplemente debido a su escala.

    Args:
        dataset (DataFrame): Dataset amb dades numèriques i categòriques.

    Returns:
        final_dataset: DataFrame amb dades numèriques i les categòriques codificades.
    """
    scaler = StandardScaler()

    # Sel·leccionem les columnes numèriques float
    numerical_columns = dataset.select_dtypes(include=['float64','int64']).columns

    # Escalem
    dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])  # algunas características con rangos grandes (como la temperatura o tiempo en milisegundos) pueden dominar la predicción
    
    print(dataset[numerical_columns].head())

    return dataset

# MAIN #########################################################################################################################
# Llegim el dataset
cleaned_dataset = pd.read_pickle(PICKLE_PATH)

# El codifiquem
scaled_dataset = escalar(cleaned_dataset)

# El convertim en pickle
scaled_dataset.to_pickle(OUT_PICKLE_PATH)

# El convertim en excel per comprovar el resultat
# output_path = 'data/codif_dataset.xlsx'  # Substitueix per la ruta de sortida
# codificated_dataset.to_excel(output_path, index=False)
