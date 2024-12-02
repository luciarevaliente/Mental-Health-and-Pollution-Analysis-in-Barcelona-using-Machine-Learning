"""
Script per analitzar dades de salut mental i contaminació de BCN
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 02/12/24
Descripció: Aquest script carrega, processa i analitza dades de salut mental i contaminació.
"""
# 02_exploratory_analysis.py
# IMPORTACIÓ
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# VARIABLES CONSTANTS
PICKLE_PATH = 'data/dataset.pkl'

# Cargar el DataFrame desde el archivo pickle guardado previamente
df = pd.read_pickle(PICKLE_PATH)


"""POR REVISAR"""

# Verificar los tipos de datos y las primeras filas
print("Tipos de datos y las primeras filas del DataFrame:")
print(df.info())

# Descripción estadística de las columnas numéricas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Revisar la cantidad de valores nulos por columna
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Visualización de la correlación entre variables numéricas
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Correlación entre Variables')
plt.savefig('results/visuals/correlation_heatmap.png')  # Guardar la imagen
plt.show()

# Visualización de la distribución de valores en una columna de ejemplo (por ejemplo, 'estres')
plt.figure(figsize=(8, 6))
sns.histplot(df['estres'].dropna(), kde=True, bins=30)
plt.title('Distribución de Estrés')
plt.savefig('results/visuals/estres_distribution.png')  # Guardar la imagen
plt.show()
