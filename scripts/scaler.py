# IMPORTACIÓ
from sklearn.preprocessing import StandardScaler
import pandas as pd

# VARIABLES
PICKLE_PATH = 'data/codif_dataset.pkl'
OUT_PICKLE_PATH = 'data/scaled_dataset.pkl'

# ESCALAT DE DADES NUMÈRIQUES ####################################################################################
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

# MAIN #########################################################################################################################
if __name__ == "__main__":
    # Leer el dataset
    cleaned_dataset = pd.read_pickle(PICKLE_PATH)

    # Escalar los datos
    scaled_dataset = escalar(cleaned_dataset)

    # Guardar el dataset escalado en formato pickle
    scaled_dataset.to_pickle(OUT_PICKLE_PATH)
    print(f"Dataset escalado guardado en {OUT_PICKLE_PATH}.")

    # Guardar en Excel para verificación (opcional)
    excel_output_path = 'data/scaled_dataset.xlsx'
    scaled_dataset.to_excel(excel_output_path, index=False)
    print(f"Dataset escalado guardado en {excel_output_path}.")
