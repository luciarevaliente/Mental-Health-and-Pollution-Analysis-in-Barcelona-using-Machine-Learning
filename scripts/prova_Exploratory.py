import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# VARIABLES CONSTANTS
PICKLE_PATH = 'data/dataset.pkl'

# Cargar el DataFrame desde el archivo pickle guardado previamente
df = pd.read_pickle(PICKLE_PATH)


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
# Filtrar només les columnes numèriques
numeric_df = df.select_dtypes(include=[float, int])

# Calcular la matriu de correlació només amb columnes numèriques
# Generar la matriz de correlación
correlation_matrix = numeric_df.corr()

# Configurar el tamaño de la figura y estilo
plt.figure(figsize=(20, 20))  # Incrementa el tamaño de la figura
sns.set(style="whitegrid")

# Crear el heatmap
sns.heatmap(
    correlation_matrix,
    annot=False,  # Si quieres valores, cambia a True
    cmap='coolwarm',
    cbar=True,
    square=True,
    xticklabels=True,
    yticklabels=True,
    linewidths=0.5
)

# Ajustar etiquetas
plt.xticks(rotation=90, fontsize=8)  # Rota y ajusta el tamaño de las etiquetas del eje X
plt.yticks(fontsize=8)  # Ajusta el tamaño de las etiquetas del eje Y
plt.title('Matriu de correlació entre variables numèriques', fontsize=15)

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
