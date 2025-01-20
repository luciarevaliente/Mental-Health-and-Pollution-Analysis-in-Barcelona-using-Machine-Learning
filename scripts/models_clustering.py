"""
Script per modelar la classe de Clustering.
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 12/12/24
Descripció: Aquest script carrega permet inicialitzar diversos models de clústering, amb funcionalitats que s'expliquen al llarg del codi.
"""
# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
import os

# CLASSE
class ClusteringModel:
    def __init__(self, data, n_clusters=3, algorithm='kmeans'):
        """
        Constructor de la classe per modelar el clustering.

        :param data: DataFrame o ndarray amb les dades per al clustering.
        :param n_clusters: Nombre de clusters a identificar. Per defecte, 3.
        :param algorithm: Algorisme de clustering a utilitzar. Pot ser 'kmeans', 'spectral', 'agglo', 'gmm'.
        """
        self.data = data
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.model = None
        self.labels = None

    def fit(self):
        """Entrenar el model segons l'algoritme seleccionat"""
        if self.algorithm == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.algorithm == 'agglo':
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        elif self.algorithm == 'gmm':
            self.model = GaussianMixture(n_components=self.n_clusters, random_state=42)
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

    def get_labels(self):
        """Obtenir les etiquetes de les prediccions"""
        return self.labels

    def get_k(self):
        return self.n_clusters

    def set_clusters(self, n_clusters):
        """
        Establir el nombre de clusters.
        
        :param n_clusters: int.
        """
        self.n_clusters = n_clusters
        
    def set_algorithm(self, algorithm):
        """
        Establir l'algoritme de clustering.
        
        :param algorithm: string que determina l'algorisme de clústering a emprar.
        """
        self.algorithm = algorithm

    # CLUSTER SELECTION ------------------------------------------------------------------------------------
    def elbow_method(self, max_clusters=15):
        """
        Aplica el mètode del colze per trobar el millor nombre de clusters per a l'algorisme Kmeans.

        :param max_clusters: int per determinar les iteracions màximes del mètode del colze
        :return: int. Valor K òptim
        """
        if self.algorithm != 'kmeans':
            raise ValueError(f"Algoritme '{self.algorithm}' no és vàlid.")

        model = KMeans(random_state=42)

        visualizer = KElbowVisualizer(model, k=(1, max_clusters))  # Provar entre 1 i max_clusters clusters

        visualizer.fit(self.data)  # Ajustar el visualitzador a les dades escalades
        visualizer.show()

        # self.n_clusters = visualizer.elbow_value_  # Obtenir el nombre òptim de clusters 
        print(f"El número òptim de clusters és: {self.n_clusters}")

        return visualizer.elbow_value_
    
    
    def silhouette(self, max_clusters=15):
        """
        Troba el millor nombre de clusters per a l'algorisme Agglomerative utilitzant l'Índex de Silueta.

        :param max_clusters: Nombre màxim de clusters a provar (per defecte, 10).
        :return: El nombre òptim de clusters segons l'índex de silueta.
        """
        if self.algorithm != 'agglo':
            raise ValueError(f"Algoritme '{self.algorithm}' no és vàlid.")

        best_silhouette = -1  # Inicialitzar la millor puntuació de silueta
        best_k = 2  # El mínim nombre de clusters per a silueta és 2

        for k in range(2, max_clusters + 1):  # Provar entre 2 i max_clusters
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(self.data)

            # Calcular l'índex de silueta
            silhouette_avg = silhouette_score(self.data, labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_k = k

        print(f"El millor K segons l'índex de silueta: {best_k}")

        return best_k


    def bic(self, max_clusters=15):
        """
        Troba el millor nombre de clusters per a l'algorisme GMM, utilitzant el BIC (Criteri d'Informació Bayesià).

        :param max_clusters: Nombre màxim de clusters a avaluar (per defecte, 10).
        :return: El nombre òptim de clusters segons el BIC.
        """
        if self.algorithm != 'gmm':
            raise ValueError(f"Algoritme '{self.algorithm}' no és vàlid.")

        bic_scores = []
        
        # Avaluar models GMM per a cada nombre de clusters des d'1 fins a max_clusters
        for k in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(self.data)
            bic_scores.append(gmm.bic(self.data))  # Calcular el BIC per al model ajustat

        # Trobar el nombre de clusters que minimitza el BIC
        best_k = np.argmin(bic_scores) + 1  # +1 perquè el rang comença en 1
        print(f"El nombre òptim de clusters segons el BIC és: {best_k}")

        return best_k


    def best_k(self, max_clusters=15):
        """
        Calcula el millor nombre de clusters per als tres mètodes (K-means, Agglomerative i Gaussian Mixture)
        i retorna el valor òptim de k.

        :param max_clusters: Nombre màxim de clusters a provar (per defecte, 10).
        :return: int. El nombre òptim de clusters segons els tres mètodes.
        """
        if self.algorithm == 'kmeans':
            # 1. Mètode del "Elbow" (K-means)
            metode = 'elbow'
            best_k = self.elbow_method(max_clusters=max_clusters)

        elif self.algorithm == 'agglo':
            # 2. Mètode de l'Índex de Silueta (Agglomerative Clustering)
            metode = 'shilouette'
            best_k = self.silhouette(max_clusters=max_clusters)

        elif self.algorithm == 'gmm':
            # 3. Criteri de Versemblança (Gaussian Mixture Model)
            metode = 'bic'
            best_k = self.bic(max_clusters=max_clusters)

        else:
            raise ValueError(f"Algoritme '{self.algorithm}' no és vàlid.")

        # Actualitzar el nombre de clústers
        self.n_clusters = best_k  

        return best_k


    # VISUALITZACIÓ -----------------------------------------------------------------------------
    def plot_clusters_PCA_2d(self):
        """
        Visualitza les dades en un gràfic 2D acolorides segons el clúster.
        Utilitza PCA per reduir la dimensionalitat si hi ha més de 2 característiques.
        """
        if self.labels is None:
            raise ValueError("El model ha de ser ajustat abans de graficar els clusters.")
        
        # Reduir a 2 dimensions si és necessari
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
        plt.title(f'Clústers Visualitzats ({self.algorithm}). PCA 2D.k={self.n_clusters}.')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

    def plot_clusters_PCA_3d(self):
        """
        Visualitza les dades en un gràfic 3D acolorides segons el clúster.
        Utilitza PCA per reduir la dimensionalitat si hi ha més de 3 característiques.
        """
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
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        fig.colorbar(scatter, label='Cluster')
        plt.show()
    
    def plot_clusters_TSNE_2d(self):
        """
        Visualitza les dades en un gràfic 2D acolorides segons el clúster.
        Utilitza TSNE per reduir la dimensionalitat si hi ha més de 2 característiques.
        """
        # Reduir a 2 dimensions si és necessari
        if self.data.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_data[:, 0], reduced_data[:, 1], 
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        plt.title(f'Clusters visualitzats ({self.algorithm}). TSNE 2D. k={self.n_clusters}.')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

    def plot_clusters_TSNE_3d(self):
        """
        Visualitza les dades en un gràfic 3D acolorides segons el clúster.
        Utilitza TSNE per reduir la dimensionalitat si hi ha més de 3 característiques.
        """
        # Reduir a 3 dimensions si és necessari
        if self.data.shape[1] > 3:
            tsne = TSNE(n_components=3, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear el gràfic
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Graficar les dades en 3D
        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        ax.set_title(f'Clusters visualitzats ({self.algorithm}). TSNE 3D. k={self.n_clusters}.')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        fig.colorbar(scatter, label='Cluster')
        plt.show()

        return reduced_data

    def plot_clusters_TSNE_3d_animated(self, filename="clusters_3d.gif"):
        """
        Visualitza les dades en un gràfic 3D acolorides segons el clúster.
        Utilitza TSNE per reduir la dimensionalitat si hi ha més de 3 característiques i guarda un GIF.

        :params filename: string. Arxiu on es guardarà el GIF.
        :return reduced_data: components obtinguts amb TSNE.
        """
        if self.labels is None:
            raise ValueError("El model ha de ser ajustat abans de graficar els clusters.")

        # Reduir a 3 dimensions si és necessari
        if self.data.shape[1] > 3:
            tsne = TSNE(n_components=3, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data

        # Crear el gràfic 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Graficar les dades en 3D
        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        ax.set_title(f'Clusters Visualitzats ({self.algorithm}). TSNE 3D. k={self.n_clusters}.')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

        # Funció per actualitzar l'animació (rotació del gràfic)
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return fig,

        # Crear animació
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

        # Guardar com a GIF
        ani.save(filename, writer='pillow', fps=20)
        plt.close(fig)
        print(f"Animació guardada com a {filename}")

        return reduced_data

    # ANÀLISI DE LES CARACTERÍSTIQUES DELS CENTROIDES -------------------------------------------------------------------------------------
    def analisi_components_centroides(self, preprocessed_df):
        """
        Analitza les mitjanes dels clústers per a cada característica en funció de l'algorisme de clustering.

        Paràmetres:
        - preprocessed_df: DataFrame amb les característiques originals (no reduïdes) utilitzades per al clustering.

        Retorna:
        - centroides_df: DataFrame amb els centres o mitjanes dels clústers per característic.
        """
        if self.labels is None or self.model is None:
            raise ValueError("El model ha de ser ajustat abans d'executar aquesta anàlisi.")

        if self.algorithm == 'kmeans':
            # KMeans: Accés directe als centroides
            if not hasattr(self.model, 'cluster_centers_'):
                raise ValueError("El model KMeans no té els centroides calculats.")
            centroids = self.model.cluster_centers_

        elif self.algorithm == 'agglo':
            # AgglomerativeClustering: Calcular les mitjanes de les característiques per cada clúster
            clusters = pd.DataFrame(self.data, columns=preprocessed_df.columns)
            clusters['Cluster'] = self.labels
            centroids = clusters.groupby('Cluster').mean().values

        elif self.algorithm == 'gmm':
            # GaussianMixture: Utilitzar les mitjanes calculades
            if not hasattr(self.model, 'means_'):
                raise ValueError("El model GaussianMixture no té les mitjanes calculades.")
            centroids = self.model.means_

        else:
            raise ValueError("Aquesta funció només admet els algorismes 'kmeans', 'agglo' o 'gmm'.")

        # Convertir els centroides a un DataFrame per una millor interpretació
        columns = preprocessed_df.columns  # Obtenir els noms de les característiques
        centroides_df = pd.DataFrame(centroids, columns=columns)
        centroides_df.index = [f"Cluster {i}" for i in range(len(centroides_df))]

        return centroides_df