import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



data = 'data/codif_dataset.pkl'
dataset = pd.read_pickle(data)
# Seleccionar las columnas relevantes (variables predictoras y la variable objetivo)
variables_predictoras = ["bienestar", "ordenador", "otrofactor"]  # Reemplaza con tus variables
import pandas as pd

# Cargar el dataset
cleaned_dataset_path = 'data/cleaned_dataset.pkl'
codif_dataset_path = 'data/codif_dataset.pkl'

cleaned_dataset = pd.read_pickle(cleaned_dataset_path)
codif_dataset = pd.read_pickle(codif_dataset_path)

# Verificar si la columna 'estres' está en el dataset limpio
if 'estres' in cleaned_dataset.columns:
    # Añadir la columna 'estres' al dataset codificado
    if cleaned_dataset.shape[0] == codif_dataset.shape[0]:
        codif_dataset['estres'] = cleaned_dataset['estres']
        print("Columna 'estres' añadida correctamente.")
    else:
        print("Error: Los datasets no tienen el mismo número de filas.")

# Verificar si 'estres' está en las columnas

scaler = MinMaxScaler()
codif_dataset['estres_scaled'] = scaler.fit_transform(codif_dataset[['estres']])
for var in variables_predictoras:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=codif_dataset[var], y=codif_dataset["estres"])
    plt.title(f"Relación entre {var} y estrés")
    plt.xlabel(var)
    plt.ylabel("Estrés")
    plt.grid(True)
    plt.show()
