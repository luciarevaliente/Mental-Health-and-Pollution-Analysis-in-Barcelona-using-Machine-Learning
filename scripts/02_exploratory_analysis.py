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


# Configuració de l'estil de visualització
sns.set(style="whitegrid")

# 1. Distribució de registres per districtes
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='district', order=df['district'].value_counts().index)
plt.title('Distribució de registres per districtes')
plt.xlabel('Nombre de registres')
plt.ylabel('Districtes')
plt.show()

# 2. Visualització de valors nuls per variable
plt.figure(figsize=(12, 6))
nulls = df.isnull().mean() * 100
nulls.sort_values(ascending=False).plot(kind='bar')
plt.title('Percentatge de valors nuls per variable')
plt.ylabel('Percentatge de valors nuls (%)')
plt.xlabel('Variables')
plt.show()

# 3. Boxplot de la variable "estres" per analitzar valors atípics
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='estres')
plt.title('Distribució de la variable "estres"')
plt.xlabel('Nivell d\'estrès')
plt.show()

# 4. Relació entre contaminació (no2bcn_24h) i salut mental (estres)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='no2bcn_24h', y='estres', alpha=0.6)
plt.title('Relació entre NO2 i estrès')
plt.xlabel('NO2 (24h)')
plt.ylabel('Nivell d\'estrès')
plt.show()

# 5. Correlació entre variables numèriques
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title('Matriz de correlació entre variables numèriques')
plt.show()

# 6. Distribució de variables de contaminació (exemple: no2bcn_24h i pm25bcn)
plt.figure(figsize=(10, 6))
sns.histplot(df['no2bcn_24h'], kde=True, color='blue', bins=30, label='NO2')
sns.histplot(df['pm25bcn'], kde=True, color='green', bins=30, label='PM2.5')
plt.title('Distribució de contaminació (NO2 i PM2.5)')
plt.xlabel('Concentració')
plt.ylabel('Freqüència')
plt.legend()
plt.show()

# 7. Comparació de "sueno" (son) per districte
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='district', y='sueno')
plt.title('Distribució de les hores de son per districte')
plt.xlabel('Districtes')
plt.ylabel('Hores de son')
plt.xticks(rotation=45)
plt.show()

# 8. Percentatge de valors nuls en variables clau
variables_clau = ['estres', 'energia', 'bienestar', 'no2bcn_24h', 'pm25bcn']
nuls_clau = df[variables_clau].isnull().mean() * 100
plt.figure(figsize=(10, 6))
nuls_clau.sort_values().plot(kind='barh', color='orange')
plt.title('Percentatge de valors nuls en variables clau')
plt.xlabel('Percentatge de valors nuls (%)')
plt.ylabel('Variables')
plt.show()
