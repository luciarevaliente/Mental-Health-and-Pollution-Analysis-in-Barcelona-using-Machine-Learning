# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# CLASSE
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
        Aplica el mètode del codo per trobar el millor nombre de clusters per KMeans i estableix en n_clusters el millor k
        
        :param max_clusters: El màxim nombre de clusters a provar per al mètode del codo.
        :param grafica: Si és True, mostra la gràfica del codo.
        """
        if self.algorithm != 'kmeans':
            raise ValueError("El mètode del codo només és aplicable per a KMeans.")
        
        # Crear el model KMeans
        kmeans = KMeans(random_state=42)
        
        # Crear el visualizador del codo (Elbow)
        visualizer = KElbowVisualizer(kmeans, k=(1, max_clusters))  # Evaluar entre 1 i max_clusters clusters
        
        # Ajustar el visualizador al dataset
        visualizer.fit(self.data)
        
        # Mostrar la gràfica només si grafica és True
        visualizer.show()
        
        # El número òptim de clusters es obté directament
        self.n_clusters = visualizer.elbow_value_
        print(f"El número òptim de clusters és: {visualizer.elbow_value_}")

    def plot_clusters_PCA_2d(self):
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
        plt.title(f'Clusters Visualitzats ({self.algorithm}). PCA 2D.k={self.n_clusters}.')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

    def plot_clusters_PCA_3d(self):
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
        ax.set_title(f'Clusters segons {self.algorithm}. PCA 3D.. k={self.n_clusters}.')
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')
        fig.colorbar(scatter, label='Cluster')
        plt.show()
    
    def plot_clusters_TSNE_2d(self):
        """
        Visualiza los datos en un gráfico 2D coloreados según el cluster.
        Utiliza t-SNE para reducir la dimensionalidad si hay más de 2 características.
        """
        if self.labels is None:
            raise ValueError("El modelo debe ser ajustado antes de graficar los clusters.")
        
        # Reducir a 2 dimensiones si es necesario
        if self.data.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear el scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_data[:, 0], reduced_data[:, 1], 
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        plt.title(f'Clusters Visualizados ({self.algorithm}). TSNE 2D. k={self.n_clusters}.')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

    def plot_clusters_TSNE_3d(self):
        """
        Visualiza los datos en un gráfico 3D coloreados según el cluster.
        Utiliza t-SNE para reducir la dimensionalidad si hay más de 3 características.
        """
        if self.labels is None:
            raise ValueError("El modelo debe ser ajustado antes de graficar los clusters.")
        
        # Reducir a 3 dimensiones si es necesario
        if self.data.shape[1] > 3:
            tsne = TSNE(n_components=3, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear el gráfico 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Graficar los datos en 3D
        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        ax.set_title(f'Clusters Visualizados ({self.algorithm}). TSNE 3D. k={self.n_clusters}.')
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')
        fig.colorbar(scatter, label='Cluster')
        plt.show()

        return reduced_data
    
    def analisi_components_centroides(self):
        pass

    def analisi_components_tsne_correlacio(self, reduced_data):
        """
        Realiza un análisis de correlación entre las características originales del conjunto de datos
        y las componentes obtenidas mediante reducción dimensional con t-SNE. Esta función calcula 
        las correlaciones entre cada componente de t-SNE y las variables originales, y devuelve una 
        lista de las 5 características con las correlaciones más fuertes (tanto positivas como negativas) 
        para cada componente.

        Parámetros:
        - reduced_data: ndarray o DataFrame que contiene los datos reducidos a través de t-SNE.
        Debe tener el mismo número de filas que self.data, y debe contener las componentes reducidas 
        generadas por t-SNE.

        Retorna:
        - dic_correlacions: un diccionario que contiene, para cada componente de t-SNE, las 5 variables 
        originales con las correlaciones más altas (positivas y negativas). La estructura es la siguiente:
        {
            'Component 1': [top_5_positive, top_5_negative],
            'Component 2': [top_5_positive, top_5_negative],
            'Component 3': [top_5_positive, top_5_negative],
        }
        Donde top_5_positive y top_5_negative son listas de las 5 características más correlacionadas
        de forma positiva y negativa, respectivamente.
        """
        COMPONENTS = ["Component 1", "Component 2", "Component 3"]

        # Convertir las coordenadas a un DataFrame
        tsne_df = pd.DataFrame(reduced_data, columns=COMPONENTS)

        # Analizar correlaciones
        combined_df = pd.concat([self.data, tsne_df], axis=1)
        correlations = combined_df.corr()[COMPONENTS]
       
        # Mostrar las correlaciones significativas
        dic_correlacions = {}

        for comp in correlations.columns:
            correlacions = correlations[comp].drop(labels=[comp])

            # print(f"Correlaciones para {comp}")
            top_5_positive = correlacions.sort_values(ascending=False).head(5)
            top_5_negative = correlacions.sort_values(ascending=True).head(5)

            dic_correlacions[comp] = [top_5_positive, top_5_negative]

        return dic_correlacions
        