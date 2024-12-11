"""
Antes:
    1. Eliminar valores null, escalar los datos y seleccionar las variables.
    2. Determinar número óptimo de clusters
    3. Aplicar algoritmo de clústering
"""
# IMPORTACIÓ
import pandas as pd
from scipy.stats import shapiro
import numpy as np

# VARIABLES CONSTANTS
PATH_DATASET = "data/scaled_dataset.pkl"
# PATH_DATASET = "data/complete_scaled_dataset.pkl"
# PATH_DATASET = "data/shuffled_scaled_dataset.pkl"


# FUNCIONS
def estudiar_normalitat(df):
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

# CLASSE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer

class ClusteringModel:
    def __init__(self, data, n_clusters=3, algorithm='kmeans'):
        """
        Constructor de la classe per a modelar el clustering.
        
        :param data: DataFrame o ndarray amb les dades a clustering.
        :param n_clusters: Nombre de clusters a identificar. Per defecte 3.
        :param algorithm: Algoritme de clustering a utilitzar. Pot ser 'kmeans', 'spectral', 'agglo', 'gmm'.
        """
        self.data = data
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.model = None
        self.labels = None

    def fit(self):
        """Entrenar el model segons l'algoritme seleccionat"""
        if self.algorithm == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters)
        elif self.algorithm == 'spectral':
            self.model = SpectralClustering(n_clusters=self.n_clusters, affinity='nearest_neighbors')
        elif self.algorithm == 'agglo':
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        elif self.algorithm == 'gmm':
            self.model = GaussianMixture(n_components=self.n_clusters)
        else:
            raise ValueError(f'Algoritme {self.algorithm} no reconegut')
        
        # Ajustar el model a les dades
        self.model.fit(self.data)
        
        # Assignar les etiquetes (clusters) al conjunt de dades
        if self.algorithm == 'gmm':
            # En el cas de GaussianMixture, s'ha d'utilitzar predict per obtenir les etiquetes
            self.labels = self.model.predict(self.data)
        else:
            self.labels = self.model.labels_

    def silhouette_score(self):
        """Calcular la puntuació de silueta per mesurar la qualitat del clustering"""
        return silhouette_score(self.data, self.labels)

    def get_labels(self):
        """Obtenir les etiquetes de les prediccions"""
        return self.labels

    def set_clusters(self, n_clusters):
        """Establir el nombre de clusters"""
        self.n_clusters = n_clusters
        
    def set_algorithm(self, algorithm):
        """Establir l'algoritme de clustering"""
        self.algorithm = algorithm

    def evaluate(self):
        """Evaluar el model amb la puntuació de silueta"""
        score = self.silhouette_score()
        print(f"Puntuació de silueta per {self.algorithm}: {score}")
        return score
    
    def elbow_method(self, max_clusters=10):
        """
        Aplica el mètode del codo per trobar el millor nombre de clusters per KMeans.
        
        :param max_clusters: El màxim nombre de clusters a provar per al mètode del codo.
        """
        if self.algorithm != 'kmeans':
            raise ValueError("El mètode del codo només és aplicable per a KMeans.")
        
        # Crear el model KMeans
        kmeans = KMeans(random_state=42)
        
        # Crear el visualizador del codo (Elbow)
        visualizer = KElbowVisualizer(kmeans, k=(1, max_clusters))  # Evaluar entre 1 y max_clusters clusters
        
        # Ajustar el visualizador al dataset
        visualizer.fit(self.data)
        
        # Mostrar la gráfica del codo
        visualizer.show()
        
        # El número óptimo de clusters se obtiene directamente
        print(f"El número óptimo de clusters es: {visualizer.elbow_value_}")
        return visualizer.elbow_value_

    def plot_clusters(self):
        """
        Visualiza los datos en un gráfico 2D coloreados según el cluster.
        Utiliza PCA para reducir la dimensionalidad si hay más de 2 características.
        """
        if self.labels is None:
            raise ValueError("El modelo debe ser ajustado antes de graficar los clusters.")
        
        # Reducir a 2 dimensiones si es necesario
        if self.data.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear el scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_data[:, 0], reduced_data[:, 1], 
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        plt.title(f'Clusters Visualitzats ({self.algorithm})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

    def plot_clusters_3d(self):
        """Visualitzar els clusters en un gràfic 3D"""
        # Si les dades tenen més de 3 dimensions, reduir a 3D amb PCA
        if self.data.shape[1] > 3:
            pca = PCA(n_components=3)
            reduced_data = pca.fit_transform(self.data)
        else:
            reduced_data = self.data

        # Crear el gràfic 3D amb els colors dels clusters
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
            c=self.labels, cmap='viridis', s=50, alpha=0.8
        )
        ax.set_title(f'Clusters segons {self.algorithm}')
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')
        fig.colorbar(scatter, label='Cluster')
        plt.show()


if __name__=="__main__":
    print('hola')
    # 1 i 2. Codificació i escalat --> Carreguem les dades preprocessades
    df = pd.read_pickle(PATH_DATASET)
    print('hola2')

    # 3. Sel·lecció de variables rellevants 
    col_importants = ['otrofactor', 'ordenador', 'covid_motor', 'dayoftheweek', 'bienestar','district_horta-guinardo', 'tmean_24h']
    reduced_dataset = df[col_importants]

    clustering_kmeans = ClusteringModel(df, algorithm='kmeans')
    clustering_kmeans.elbow_method(max_clusters=50)

    clustering_kmeans.fit()
    clustering_kmeans.evaluate()
    clustering_kmeans.plot_clusters()
    clustering_kmeans.plot_clusters_3d()

    # reduced_dataset['cluster'] = clustering_kmeans.get_labels() 
    # cluster_stats = reduced_dataset.groupby('cluster').mean()
    # # cluster_stats[:,0]
    # for col in cluster_stats:
    #     print(cluster_stats[col])
    
    ###############################################################################################################3

    # 4. El·lecció algoritme clustering: inicialitzar la classe ClusteringModel i provar els diferents algoritmes
    # clustering_kmeans = ClusteringModel(df, algorithm='kmeans')
    # best_k = clustering_kmeans.elbow_method(max_clusters=50)

    # clustering_kmeans = ClusteringModel(df, n_clusters=best_k, algorithm='kmeans')
    # clustering_kmeans.fit()
    # clustering_kmeans.evaluate()
    # clustering_kmeans.plot_clusters()
    # clustering_kmeans.plot_clusters_3d()
    
    # clustering_spectral = ClusteringModel(df, n_clusters=3, algorithm='spectral')
    # clustering_spectral.fit()
    # clustering_spectral.evaluate()
    
    # clustering_agglo = ClusteringModel(df, n_clusters=3, algorithm='agglo')
    # clustering_agglo.fit()
    # clustering_agglo.evaluate()
    
    # clustering_gmm = ClusteringModel(df, n_clusters=3, algorithm='gmm')
    # clustering_gmm.fit()
    # clustering_gmm.evaluate()