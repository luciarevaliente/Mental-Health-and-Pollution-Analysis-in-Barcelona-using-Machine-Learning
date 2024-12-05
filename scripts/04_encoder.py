from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
PICKLE_PATH = 'data/dataset.pkl'


data = pd.read_pickle(PICKLE_PATH)
# print(data.head)

# Reemplazar valores nulos siguiendo la probabilidad de cada columna
data_cleaned = data.copy()
for column in data_cleaned.columns: # per cada categoria
    if data_cleaned[column].isnull().sum() > 0: # si es null
        if data_cleaned[column].dtype == 'object':
            # Reemplazar valores categóricos nulos según su distribución de frecuencias
            probabilities = data_cleaned[column].value_counts(normalize=True)
            print('prob',probabilities)
            data_cleaned[column] = data_cleaned[column].apply(
                lambda x: np.random.choice(probabilities.index, p=probabilities.values) if pd.isnull(x) else x)
            # Calcula las probabilidades de frecuencia de cada categoría en la columna. Esto genera un objeto que representa las proporciones relativas de cada categoría.
        else:
            # Reemplazar valores numéricos nulos según su distribución
            valid_values = data_cleaned[column].dropna()
            data_cleaned[column] = data_cleaned[column].apply(
                lambda x: np.random.choice(valid_values) if pd.isnull(x) else x
            )
        print(data_cleaned[column])

# # Aplicar One-Hot Encoding (OHE) a las columnas categóricas
# categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
# encoder = OneHotEncoder(sparse=False, drop='first')
# categorical_encoded = encoder.fit_transform(data_cleaned[categorical_columns])
# encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# # Concatenar las columnas OHE con las demás columnas numéricas
# numerical_data = data_cleaned.select_dtypes(exclude=['object'])
# final_data = pd.concat([numerical_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# # Mostrar las primeras filas del dataset procesado
# import ace_tools as tools; tools.display_dataframe_to_user(name="Dataset Procesado para Predicción de Estrés", dataframe=final_data)

# final_data.head()
