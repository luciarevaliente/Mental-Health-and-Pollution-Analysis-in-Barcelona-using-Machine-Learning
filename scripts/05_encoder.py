from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# VARIABLES
CLEANED_PICKLE_PATH = 'data/cleaned_dataset.pkl'
CODIF_PICKLE_PATH = 'data/codif_dataset.pkl'

# CODIFICACIÓ DE LES DADES CATEGÒRIQUES
def codificacio_dades_categoriques(dataset):
    """
    Función que codifica los datos categóricos de un dataset utilizando OneHotCoder.

    Usamos OneHotEncoder ya que no introduce un orden artificial, así el modelo no interpreta un una relación de rango entre las categorías, lo cual no es 
    cierto para datos nominales. Además, OneHotEncoder es adecuado tanto para variables binarias como para nominales, ya que genera una representación 
    independiente para cada categoría.

    Args:
        dataset (DataFrame): Dataset amb dades numèriques i categòriques.

    Returns:
        final_dataset: DataFrame amb dades numèriques i les categòriques codificades.
    """
    # Seleccionamos las columnas categóricas
    categorical_columns = dataset.select_dtypes(include=['object']).columns

    # Aplicamos One-Hot Encoding a las columnas categóricas
    encoder = OneHotEncoder(#drop='first',  #Si tienes un dataset muy grande y muchas categorías en tus columnas, eliminar una categoría por variable puede reducir significativamente el número de columnas generadas, mejorando el rendimiento del modelo y ahorrando memoria.
                            sparse_output=False)  # sparse=False: resultado en forma de array y no de matriz
    
    encoded_categorical = encoder.fit_transform(dataset[categorical_columns]) # Adaptem el codificador a dades categòriques

    # Convertir el resultado codificado a un DataFrame
    encoded_dataset = pd.DataFrame(encoded_categorical, # Convertim l'array codificat en un nou dataset
            columns = encoder.get_feature_names_out(categorical_columns),  # Generem les noves columnes codificades amb el format: columnaOriginal_categoria
            index = dataset.index  # Ens assegurem que les files coincideixin
    )

    # Unir el DataFrame codificado con las columnas numéricas
    final_dataset = pd.concat(
        [dataset.drop(columns=categorical_columns), encoded_dataset],  # Eliminem les dades categòriques inicials i afegim les codificades
        axis=1  # Indiquem que són columnes
    )
    return final_dataset

# MAIN
# Llegim el dataset
cleaned_dataset = pd.read_pickle(CLEANED_PICKLE_PATH)

# El codifiquem
codificated_dataset = codificacio_dades_categoriques(cleaned_dataset)

# El convertim en pickle
codificated_dataset.to_pickle(CODIF_PICKLE_PATH)

# El convertim en excel per comprovar el resultat
# output_path = 'data/codif_dataset.xlsx'  # Substitueix per la ruta de sortida
# codificated_dataset.to_excel(output_path, index=False)
