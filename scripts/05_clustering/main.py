"""
Antes:
    1. Eliminar valores null, escalar los datos y seleccionar las variables.
    2. Determinar número óptimo de clusters
    3. Aplicar algoritmo de clústering
"""

import pandas as pd
from scipy.stats import shapiro

# Supongamos que tu DataFrame se llama 'df'
# Carga tus datos si están en un archivo CSV o similar
df = pd.read_pickle("data/cleaned_dataset.pkl")

# Aplicar Shapiro-Wilk a cada columna numérica
resultados = {}
for columna in df.select_dtypes(include=['float64', 'int64']).columns:  # Solo columnas numéricas
    stat, p_value = shapiro(df[columna].dropna())  # Ignora valores NaN
    resultados[columna] = {"Estadístico W": stat, "Valor p": p_value}

# Mostrar resultados
for columna, res in resultados.items():
    print(f"Columna: {columna}")
    print(f"  Estadístico W: {res['Estadístico W']:.4f}")
    print(f"  Valor p: {res['Valor p']:.4f}")
    if res["Valor p"] > 0.05:
        print("  → Los datos parecen seguir una distribución normal.")
    else:
        print("  → Los datos no siguen una distribución normal.")
