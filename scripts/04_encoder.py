from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

PICKLE_PATH = 'data/dataset.pkl'

# Cargar el dataset
data = pd.read_pickle(PICKLE_PATH)

# Copia del DataFrame
data_cleaned = data.copy()

# Eliminar columnas con demasiados valores faltantes (más del 50% de NaN)
data_cleaned = data_cleaned.loc[:, data_cleaned.isnull().mean() < 0.5]

# Usar KNNImputer para imputar valores numéricos
knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = knn_imputer.fit_transform(data_cleaned.select_dtypes(include=['number']))

# Actualizar el DataFrame con los valores imputados
data_cleaned[data_cleaned.select_dtypes(include=['number']).columns] = data_imputed

# Imputar columnas categóricas con la moda
for column in data_cleaned.select_dtypes(include=['object']).columns:
    data_cleaned[column].fillna(data_cleaned[column].mode()[0], inplace=True)

# Resultado después de la imputación
print("Dataset con valores nulos imputados:")
print(data_cleaned.head())

# # Aplicar One-Hot Encoding a las columnas categóricas
# categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
# encoder = OneHotEncoder(drop='first', sparse=False)
# encoded_categorical = encoder.fit_transform(data_cleaned[categorical_columns])

# # Convertir el resultado codificado a un DataFrame
# encoded_df = pd.DataFrame(
#     encoded_categorical,
#     columns=encoder.get_feature_names_out(categorical_columns),
#     index=data_cleaned.index
# )

# # Unir el DataFrame codificado con las columnas numéricas
# data_ohe = pd.concat(
#     [data_cleaned.drop(columns=categorical_columns), encoded_df],
#     axis=1
# )

# print("\nDataset después de One-Hot Encoding:")
# print(data_ohe.head())
