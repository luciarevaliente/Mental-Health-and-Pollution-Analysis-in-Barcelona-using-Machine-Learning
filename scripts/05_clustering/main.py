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


#PROBAR PARÁMETROS CON CLUSTERING!!!!!!!!!!!!!!11111111111111111111111111111!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset escalado
dataset_path = 'data/scaled_dataset.pkl'
scaled_dataset = pd.read_pickle(dataset_path)

# Seleccionar características para clustering (excluyendo la variable objetivo)
TARGET_COLUMN = 'estres'
X = scaled_dataset.drop(columns=[TARGET_COLUMN])

# Resultados de clustering
clustering_results = {}

# 1. K-Means Clustering
print("\n--- K-Means Clustering ---")
kmeans = KMeans(n_clusters=3, random_state=42)  # Cambia n_clusters según sea necesario
kmeans_labels = kmeans.fit_predict(X)
silhouette_kmeans = silhouette_score(X, kmeans_labels)

clustering_results['kmeans'] = {
    'labels': kmeans_labels,
    'silhouette_score': silhouette_kmeans
}
print(f"Silhouette Score (K-Means): {silhouette_kmeans}")

# 2. DBSCAN Clustering
print("\n--- DBSCAN Clustering ---")
dbscan = DBSCAN(eps=1.5, min_samples=5)  # Ajusta eps y min_samples según los datos
dbscan_labels = dbscan.fit_predict(X)
silhouette_dbscan = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

clustering_results['dbscan'] = {
    'labels': dbscan_labels,
    'silhouette_score': silhouette_dbscan
}
print(f"Silhouette Score (DBSCAN): {silhouette_dbscan}")

# 3. Agglomerative Clustering
print("\n--- Agglomerative Clustering ---")
agg_clustering = AgglomerativeClustering(n_clusters=3)  # Cambia n_clusters según sea necesario
agg_labels = agg_clustering.fit_predict(X)
silhouette_agg = silhouette_score(X, agg_labels)

clustering_results['agglomerative'] = {
    'labels': agg_labels,
    'silhouette_score': silhouette_agg
}
print(f"Silhouette Score (Agglomerative): {silhouette_agg}")

# 4. Visualización (opcional, para K-Means)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("Clustering K-Means (Primeras 2 características)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar()
plt.show()

# Guardar etiquetas de clustering
for method, results in clustering_results.items():
    scaled_dataset[f"{method}_cluster"] = results['labels']

output_path = 'data/clustered_dataset.pkl'
scaled_dataset.to_pickle(output_path)
print(f"Dataset con etiquetas de clustering guardado en {output_path}.")
