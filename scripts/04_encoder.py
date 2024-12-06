"""
Usamos OneHotEncoder ya que no introduce un orden artificial, así el modelo no interpreta un una relación de rango entre 
las categorías, lo cual no es cierto para datos nominales.

OneHotEncoder es adecuado tanto para variables binarias como para nominales, ya que genera una representación independiente para cada categoría.
"""

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# VARIABLES
PICKLE_PATH = 'data/cleaned_dataset.pkl'

# Cargamos el dataset
cleaned_dataset = pd.read_pickle(PICKLE_PATH)

# Seleccionamos las columnas categóricas
categorical_columns = cleaned_dataset.select_dtypes(include=['object']).columns

# Aplicamos One-Hot Encoding a las columnas categóricas
encoder = OneHotEncoder(#drop='first',  #Si tienes un dataset muy grande y muchas categorías en tus columnas, eliminar una categoría por variable puede reducir significativamente el número de columnas generadas, mejorando el rendimiento del modelo y ahorrando memoria.
                        sparse=False)  # sparse=False: resultado en forma de array y no de matriz
encoded_categorical = encoder.fit_transform(cleaned_dataset[categorical_columns])

# Convertir el resultado codificado a un DataFrame
encoded_df = pd.DataFrame(encoded_categorical, # Convertim l'array codificat en un nou dataset
        columns = encoder.get_feature_names_out(categorical_columns),  # Generem les noves columnes codificades amb el format: columnaOriginal_categoria
        index = cleaned_dataset.index  # Ens assegurem que les files coincideixin
)

# Unir el DataFrame codificado con las columnas numéricas
data_ohe = pd.concat(
    [cleaned_dataset.drop(columns=categorical_columns), encoded_df],  # Eliminem les dades categòriques inicials i afegim les codificades
    axis=1  # Indiquem que són columnes
)

print("\nDataset después de One-Hot Encoding:")
print(data_ohe.head())
